# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import numpy as np
from typing import Tuple

import torch
from torch import device, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom

logger = logging.getLogger(__name__)

import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
####新增IFC模块
from .ifc.models.backbone import Joiner
from .ifc.models.ifc import IFC, SetCriterion
from .ifc.models.matcher import HungarianMatcher
from .ifc.models.position_encoding import PositionEmbeddingSine
from .ifc.models.segmentation import MaskHead, segmentation_postprocess
from .ifc.models.transformer import IFCTransformer
from .ifc.structures.clip_output import Videos, Clips
from .ifc.util.misc import NestedTensor, _max_by_axis, interpolate

from .ifc.models.misc import MLP
from .ifc.ifc import Ifc

####新增IFC的MaskedBackbone
class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        # self.backbone = build_backbone(cfg)
        # backbone_shape = self.backbone.output_shape()
        # self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.feature_strides = None
        # self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images, features):
        # features = self.backbone(images.tensor)
        self.feature_strides = features['feature_strides']
        features = dict([(key, features[key]) for key in ['res2','res3','res4','res5']])
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks

@META_ARCH_REGISTRY.register()
class VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool, 
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames, 
        cfg, #IFC模块
    ):
        super().__init__()
        self.backbone = backbone
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]

        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        # self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.num_frames = num_frames
        
        #IFC模块下的
        # cfg = self.cfg
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM

        device = torch.device(cfg.MODEL.DEVICE)
        self.clip_stride = cfg.MODEL.IFC.CLIP_STRIDE
        self.merge_on_cpu = cfg.MODEL.IFC.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.IFC.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.IFC.APPLY_CLS_THRES

        self.is_coco = cfg.DATASETS.TEST[0].startswith("coco")
        self.num_classes = cfg.MODEL.IFC.NUM_CLASSES
        self.mask_stride = cfg.MODEL.IFC.MASK_STRIDE
        self.match_stride = cfg.MODEL.IFC.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.IFC.HIDDEN_DIM
        num_queries = cfg.MODEL.IFC.NUM_OBJECT_QUERIES
        self.num_queries = num_queries

        # Transformer parameters:
        nheads = cfg.MODEL.IFC.NHEADS
        dropout = cfg.MODEL.IFC.DROPOUT
        dim_feedforward = cfg.MODEL.IFC.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.IFC.ENC_LAYERS
        dec_layers_ifc = cfg.MODEL.IFC.DEC_LAYERS
        pre_norm = cfg.MODEL.IFC.PRE_NORM
        num_memory_bus = cfg.MODEL.IFC.NUM_MEMORY_BUS

        # Loss parameters:
        mask_weight_ifc = cfg.MODEL.IFC.MASK_WEIGHT
        dice_weight_ifc = cfg.MODEL.IFC.DICE_WEIGHT
        deep_supervision_ifc = cfg.MODEL.IFC.DEEP_SUPERVISION
        no_object_weight_ifc = cfg.MODEL.IFC.NO_OBJECT_WEIGHT

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone_ifc = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        # backbone_ifc.num_channels = d2_backbone.num_channels

        transformer = IFCTransformer(num_frames=self.num_frames,d_model=hidden_dim,dropout=dropout,nhead=nheads,
            num_memory_bus=num_memory_bus,dim_feedforward=dim_feedforward,num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers_ifc,normalize_before=pre_norm,return_intermediate_dec=deep_supervision_ifc,)
        mask_head = MaskHead(hidden_dim, [1024, 512], self.num_frames)
        self.detr = IFC(backbone_ifc, transformer, mask_head,num_classes=self.num_classes, num_queries=num_queries,
            num_frames=self.num_frames, aux_loss=deep_supervision_ifc)
        self.detr.to(device)
        # building criterion
        matcher_ifc = HungarianMatcher(cost_class=1,cost_dice=dice_weight_ifc,num_classes=self.num_classes,)
        weight_dict_ifc = {"loss_ce": 1, "loss_mask": mask_weight_ifc, "loss_dice": dice_weight_ifc}
        if deep_supervision_ifc:
            aux_weight_dict = {}
            for i in range(dec_layers_ifc - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict_ifc.items()})
            weight_dict_ifc.update(aux_weight_dict)
        losses = ["labels", "masks", "cardinality"]
        self.criterion_ifc = SetCriterion(self.num_classes, matcher=matcher_ifc, weight_dict=weight_dict_ifc, 
            eos_coef=no_object_weight_ifc, losses=losses,num_frames=self.num_frames)
        self.criterion_ifc.to(device)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(device)
        self.merge_device = "cpu" if self.merge_on_cpu else device

        self.size_window = cfg.MODEL.IFC.size_window

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION #Transformer Decoder每一层是否都计算loss
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "cfg": cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = []
        #视频预处理
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        #Frame-Level模型前向推理
        features = self.backbone(images.tensor) #[res2,res3,res4,res5]
        outputs = self.sem_seg_head(features)
        outputs_frame_level = dict([(key,outputs[key]) for key in ['pred_logits','pred_masks','aux_outputs']])
        
        #shift-window预处理
        object_tokens = outputs['object_tokens'] #[100, 8*5, 256] [N, B*T, C]
        del outputs
        b = len(batched_inputs)
        t = self.num_frames

        t_o = object_tokens.shape[1] #用于测试时的判断
        if b < 2 : 
            t = t_o
            
        n = self.num_queries
        c = object_tokens.shape[2]
        if b < 2 :
            object_tokens = object_tokens.unsqueeze(1)
        else:
            object_tokens = object_tokens.view(n, b, t, c)
        b = b if self.training else 1
       
        # object_tokens = object_tokens.permute(1, 2, 0, 3).flatten(2,3) # THW, B, C 
        # features['feature_strides'] = self.feature_strides
        out = []
        for i in range(0, self.size_window-1, self.size_window//2):
            for j in range(i, t-1, self.size_window):
                if t-j < self.size_window:
                    # window_tokens = object_tokens[:,:,j:,:] #[100, 8, 3, 256]
                    # mlp = MLP(256*(t-j), 256, 256, 2).to(self.device)
                    break
                elif t < self.num_frames :
                    window_tokens = object_tokens[:,:,j:,:] #[100, 8, 3, 256]
                    mlp = MLP(256*(t-j), 256, 256, 2).to(self.device)
                else:
                    window_tokens = object_tokens[:,:,j:j+self.size_window,:] #[100, 8, 3, 256]
                    mlp = MLP(256*(self.size_window), 256, 256, 2).to(self.device)
                window_tokens = window_tokens.flatten(2)
                
                window_tokens = mlp(window_tokens)
                #VITA前向推理
                # out_vita = self.detr(images, window_tokens, features)
                out_vita = self.detr(window_tokens, features)
                out.append(out_vita)
                del out_vita, window_tokens

        pred_logits = []
        pred_masks = []
        aux_outputs_pred_logits_0 = []
        aux_outputs_pred_masks_0 = []
        aux_outputs_pred_logits_1 = []
        aux_outputs_pred_masks_1 = []
        n = len(out)
        for i in range(n):
            pred_logits.append(out[i]['pred_logits'])
            pred_masks.append(out[i]['pred_masks'])
            aux_outputs_pred_logits_0.append(out[i]['aux_outputs'][0]['pred_logits'])
            aux_outputs_pred_masks_0.append(out[i]['aux_outputs'][0]['pred_masks'])
            aux_outputs_pred_logits_1.append(out[i]['aux_outputs'][1]['pred_logits'])
            aux_outputs_pred_masks_1.append(out[i]['aux_outputs'][1]['pred_masks'])
        del out
        out_vita = {'pred_logits':sum(pred_logits)/n, 'pred_masks':sum(pred_masks)/n,
            'aux_outputs':[
                            {'pred_logits':sum(aux_outputs_pred_logits_0)/2,
                            'pred_masks':sum(aux_outputs_pred_masks_0)/2},
                            {'pred_logits':sum(aux_outputs_pred_logits_1)/2,
                            'pred_masks':sum(aux_outputs_pred_masks_1)/2}
                            ]
        }
        del pred_logits,pred_masks,aux_outputs_pred_logits_0,aux_outputs_pred_masks_0,
        aux_outputs_pred_logits_1,aux_outputs_pred_masks_1

        if self.training:
            # Frame-Level模块 loss处理
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            # bipartite matching-based loss
            losses = self.criterion(outputs_frame_level, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            
            # IFC模块 loss处理
            targets_ifc = self.prepare_targets_ifc(batched_inputs)
            loss_dict = self.criterion_ifc(out_vita, targets_ifc)
            weight_dict_ifc = self.criterion_ifc.weight_dict
            for k in list(loss_dict.keys()):
                if k in weight_dict_ifc:
                    loss_dict[k] *= weight_dict_ifc[k]
                else:
                    loss_dict.pop(k)

            return {**losses, **loss_dict}  #前后模块loss相加

        else:
            mask_cls_results = out_vita["pred_logits"]
            mask_pred_results = out_vita["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            # upsample masks
            mask_pred_result = retry_if_cuda_oom(F.interpolate)( #尝试解决显存不足问题
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del out_vita

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)



    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def prepare_targets_ifc(self, targets):
        gt_instances = []

        _sizes = []
        for targets_per_video in targets:
            _sizes += [list(t.image_size) for t in targets_per_video["instances"]]
        max_size = _max_by_axis(_sizes)

        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames] + max_size
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            gt_classes_per_video = torch.full((_num_instance,), self.num_classes, device=self.device)
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                valid_cls_idx = (targets_per_frame.gt_classes != self.num_classes)
                gt_classes_per_video[valid_cls_idx] = targets_per_frame.gt_classes[valid_cls_idx]
                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            o_h, o_w = gt_masks_per_video.shape[-2:]
            l_h, l_w = math.ceil(o_h/self.mask_stride), math.ceil(o_w/self.mask_stride)
            m_h, m_w = math.ceil(o_h/self.match_stride), math.ceil(o_w/self.match_stride)

            gt_masks_for_loss  = interpolate(gt_masks_per_video, size=(l_h, l_w), mode="bilinear", align_corners=False)
            gt_masks_for_match = interpolate(gt_masks_per_video, size=(m_h, m_w), mode="bilinear", align_corners=False)
            gt_instances[-1].update({"masks": gt_masks_for_loss, "match_masks": gt_masks_for_match})

        return gt_instances

    def prepare_targets_ifc(self, targets):
        gt_instances = []

        _sizes = []
        for targets_per_video in targets:
            _sizes += [list(t.image_size) for t in targets_per_video["instances"]]
        max_size = _max_by_axis(_sizes)

        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames] + max_size
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            gt_classes_per_video = torch.full((_num_instance,), self.num_classes, device=self.device)
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                valid_cls_idx = (targets_per_frame.gt_classes != self.num_classes)
                gt_classes_per_video[valid_cls_idx] = targets_per_frame.gt_classes[valid_cls_idx]
                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            o_h, o_w = gt_masks_per_video.shape[-2:]
            l_h, l_w = math.ceil(o_h/self.mask_stride), math.ceil(o_w/self.mask_stride)
            m_h, m_w = math.ceil(o_h/self.match_stride), math.ceil(o_w/self.match_stride)

            gt_masks_for_loss  = interpolate(gt_masks_per_video, size=(l_h, l_w), mode="bilinear", align_corners=False)
            gt_masks_for_match = interpolate(gt_masks_per_video, size=(m_h, m_w), mode="bilinear", align_corners=False)
            gt_instances[-1].update({"masks": gt_masks_for_loss, "match_masks": gt_masks_for_match})

        return gt_instances

        
    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
