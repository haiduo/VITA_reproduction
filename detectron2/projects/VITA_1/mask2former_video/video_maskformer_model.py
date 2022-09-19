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

import sys,os
# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from .modeling.criterion import VideoSetCriterion,ImageSetCriterion
from .modeling.matcher import VideoHungarianMatcher,ImageHungarianMatcher
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
        cfg, #IFC模块
    ):
        super().__init__()
        self.backbone = backbone
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]

        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
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
        
        #IFC模块下的
        batch_size = cfg.SOLVER.IMS_PER_BATCH
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
        window_size = cfg.MODEL.IFC.window_size
        depth = cfg.MODEL.IFC.depth

        # Transformer parameters:
        nheads = cfg.MODEL.IFC.NHEADS
        dropout = cfg.MODEL.IFC.DROPOUT
        dim_feedforward = cfg.MODEL.IFC.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.IFC.ENC_LAYERS
        dec_layers_ifc = cfg.MODEL.IFC.DEC_LAYERS

        # Loss parameters:
        class_weight_ifc = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        mask_weight_ifc = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        dice_weight_ifc = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        deep_supervision_ifc = cfg.MODEL.IFC.DEEP_SUPERVISION
        no_object_weight_ifc = cfg.MODEL.IFC.NO_OBJECT_WEIGHT

        transformer = IFCTransformer(batch_size=batch_size,num_frames=self.num_frames, num_queries=num_queries, dim_attenion=hidden_dim,
            num_heads=nheads, window_size=window_size,
            num_encoder_layers=enc_layers, depth=depth, num_decoder_layers=dec_layers_ifc,
            dim_feedforward=dim_feedforward, dropout=dropout, return_intermediate_dec=deep_supervision_ifc,
            use_checkpoint = cfg.MODEL.IFC.USE_CHECKPOINT)
        mask_head = MaskHead(hidden_dim)
        self.detr = IFC(transformer, mask_head, num_classes=self.num_classes, num_queries=num_queries,
            num_frames=self.num_frames, aux_loss=deep_supervision_ifc)
        
        # building IFC criterion
        matcher_ifc = HungarianMatcher(cost_class=class_weight_ifc,
                            cost_dice=dice_weight_ifc, num_classes=self.num_classes,)
        weight_dict_ifc = {"loss_ce_ifc": 1, "loss_mask_ifc": mask_weight_ifc, "loss_dice_ifc": dice_weight_ifc}
        if deep_supervision_ifc:
            aux_weight_dict = {}
            for i in range(dec_layers_ifc - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict_ifc.items()})
            weight_dict_ifc.update(aux_weight_dict)
        losses = ["labels", "masks", "cardinality"]
        self.criterion_ifc = SetCriterion(self.num_classes, matcher=matcher_ifc, weight_dict=weight_dict_ifc, 
            eos_coef=no_object_weight_ifc, losses=losses,num_frames=self.num_frames)
        self.merge_device = "cpu" if self.merge_on_cpu else device
        self.to(device)

    @classmethod #https://zhuanlan.zhihu.com/p/35643573
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION #Transformer Decoder每一层是否都计算loss
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT  # λcls = 2
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT    # λdice = 5
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT    # λce = 2

        # building criterion
        matcher = ImageHungarianMatcher(  # VideoHungarianMatcher
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

        criterion = ImageSetCriterion( #VideoSetCriterion
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
            "cfg": cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
      
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device)) #视频预处理
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        #Frame-Level模型前向推理
        features = self.backbone(images.tensor) # [res2,res3,res4,res5]
        outputs_frame_level, features_queries, mask_features = self.sem_seg_head(features)

        #VITA前向推理
        out_vita = self.detr(features_queries, mask_features)

        if self.training:
            # Frame-Level模块 mask classification target loss处理 
            if "instances" in batched_inputs[0]:
                gt_instances = []
                for x in batched_inputs:
                    x = [sub_x.to(self.device) for sub_x in x["instances"]]
                    gt_instances.extend(x)
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None # mask classification target
            losses = self.criterion(outputs_frame_level, targets) # bipartite matching-based loss
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k) # remove this loss if not specified in `weight_dict`
            
            # IFC模块 loss处理
            targets_ifc = self.prepare_targets_ifc(batched_inputs) #防止迭代增加loss
            loss_dict = self.criterion_ifc(out_vita, targets_ifc)
            weight_dict_ifc = self.criterion_ifc.weight_dict
            for k in list(loss_dict.keys()):
                if k in weight_dict_ifc:
                    loss_dict[k] *= weight_dict_ifc[k]
                else:
                    loss_dict.pop(k)

            return {**losses, **loss_dict}  #前后模块loss合并
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
            del out_vita #https://blog.csdn.net/windscloud/article/details/79732014
            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation
            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])
            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)


    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks.tensor
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def prepare_targets_video(self, targets, images):
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
