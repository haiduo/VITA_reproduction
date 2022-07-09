"""
https://github.com/facebookresearch/detr
"""
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import sys, os
sys.path.append(os.getcwd())

from mask2former_video.config import *

# projects/VITA/Mask2Former/mask2former_video/config.py

class IFCTransformer(nn.Module):

    def __init__(self, num_frames, d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 num_memory_bus=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.num_frames = num_frames
        self.num_memory_bus = num_memory_bus
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = IFCEncoder(num_frames, encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.clip_decoder = IFCDecoder(decoder_layer, num_decoder_layers, num_frames, decoder_norm, return_intermediate=return_intermediate_dec)
        self._reset_parameters() #默认初始化
        self.return_intermediate_dec = return_intermediate_dec
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pad_zero(self, x, pad, dim=0):
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    def forward(self, window_tokens, query_embed, is_train):
        # prepare for enc-dec
        # bs = src.shape[0] // self.num_frames if is_train else 1
        # t = src.shape[0] // bs
        # _, c, h, w = src.shape

        # encoder
        # src = src.view(bs*t, c, h*w).permute(2, 0, 1)               # HW, BT, C
        # frame_pos = pos_embed.view(bs*t, c, h*w).permute(2, 0, 1)   # HW, BT, C
        # frame_mask = mask.view(bs*t, h*w)                           # BT, HW

        src = self.encoder(window_tokens, is_train=is_train)

        # decoder  #*****此处与之前Mask2Former交接********
        # t = size_window
        # bs = window_tokens.shape[1]//t
        # c = window_tokens.shape[2]
        # n = window_tokens.shape[0]
        # dec_src = src.view(h*w, bs, t, c).permute(2, 0, 1, 3).flatten(0,1) # THW, B, C 
        query_embed = query_embed.unsqueeze(1).repeat(1, window_tokens.shape[1], 1)     # N, B, C
        tgt = torch.zeros_like(query_embed)

        # dec_pos = pos_embed.view(bs, t, c, h*w).permute(1, 3, 0, 2).flatten(0,1) # THW, B, C 
        # dec_mask = mask.view(bs, t*h*w)                             # B, THW
        #经过三层transformer decoder
        clip_hs = self.clip_decoder(tgt, src, query_pos=query_embed, is_train=is_train) #[3, 100, 8, 256]

        # ret_memory = src.permute(1,2,0).reshape(bs*t, c, h, w)

        return clip_hs


class IFCEncoder(nn.Module):

    def __init__(self, num_frames, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.enc_layers = _get_clones(encoder_layer, num_layers)
        self.bus_layers = _get_clones(encoder_layer, num_layers)
        norm = [copy.deepcopy(norm) for i in range(2)]
        self.out_norm, self.bus_norm = norm

    def pad_zero(self, x, pad, dim=0):
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    def forward(self, output, 
                mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, 
                is_train: bool = True):
        # bs = src.shape[1] // self.num_frames if is_train else 1
        # t = src.shape[1] // bs
        # hw, _, c = src.shape
        # M = len(memory_bus)

        # memory_bus = memory_bus[:, None, :].repeat(1, bs*t, 1)
        # memory_pos = memory_pos[:, None, :].repeat(1, bs*t, 1)

        # pos = torch.cat((pos, memory_pos))
        # mask = self.pad_zero(mask, dim=1)
        # src_key_padding_mask = self.pad_zero(src_key_padding_mask, dim=1)

        # output = src

        for layer_idx in range(self.num_layers):
            # output = torch.cat((output, memory_bus))
            output = self.enc_layers[layer_idx](output, src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            # output, memory_bus = output[:hw, :, :], output[hw:, :, :]

            # memory_bus = memory_bus.view(M, bs, t, c).permute(2,1,0,3).flatten(1,2) # TxBMxC
            # memory_bus = self.bus_layers[layer_idx](memory_bus)
            # memory_bus = memory_bus.view(t, bs, M, c).permute(2,1,0,3).flatten(1,2) # MxBTxC

        if self.out_norm is not None:
            output = self.out_norm(output)
        # if self.bus_norm is not None:
        #     memory_bus = self.bus_norm(memory_bus)

        return output


class IFCDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, num_frames, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.norm = norm
        self.return_intermediate = return_intermediate

    def pad_zero(self, x, pad, dim=0):
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    def forward(self, tgt, dec_src, #tgt为object Decoder的输入video query，而dec_src为k与v
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                is_train: bool = True):
        output = tgt  #相当于 Object queries
        return_intermediate = (self.return_intermediate and is_train)
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, dec_src, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2) #残差连接
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = self.with_pos_embed(tgt, query_pos) #video queries
        k = v = memory #object Tokens
        tgt2 = self.self_attn(q, k, value=v)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
