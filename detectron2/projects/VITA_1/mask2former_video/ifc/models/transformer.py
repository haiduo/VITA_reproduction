"""
https://github.com/facebookresearch/detr
"""
import copy
from curses import window
from turtle import shape
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import sys, os
sys.path.append(os.getcwd())

from mask2former_video.config import *
from timm.models.layers import DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint
import numpy as np

# projects/VITA/Mask2Former/mask2former_video/config.py


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class IFCTransformer(nn.Module):

    def __init__(self, batch_size, num_frames, num_queries=100, dim_attenion=256, 
                num_heads=8, window_size=6, 
                num_encoder_layers=3, depth=2, num_decoder_layers=3,
                dim_feedforward=2048, dropout=0.1, return_intermediate_dec=False,
                activation="relu", mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0., drop_path=0.,
                use_checkpoint=False
        ):
        super().__init__()
        self.num_frames = num_frames
        encoder_layer = nn.ModuleList([
            SwinTransformerBlock(batch_size=batch_size, num_queries=num_queries, dim_attenion=dim_attenion, 
                                num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2, 
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                dropout=dropout, attn_drop=attn_drop, 
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                                act_layer=nn.GELU, norm_layer=nn.LayerNorm
                                )
            for i in range(depth)])
        
        self.encoder = IFCEncoder(num_frames, num_queries, window_size,
                encoder_layer, num_layers=num_encoder_layers, use_checkpoint = use_checkpoint,)
        decoder_layer = TransformerDecoderLayer(dim_attenion, num_heads, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(dim_attenion)
        self.clip_decoder = IFCDecoder(decoder_layer, num_decoder_layers, decoder_norm, 
                                        return_intermediate=return_intermediate_dec)
        self._reset_parameters() #默认初始化
        self.return_intermediate_dec = return_intermediate_dec
        self.dim_attenion = dim_attenion
        self.num_heads = num_heads

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
        # object encoder
        src = self.encoder(window_tokens,)
        N, BT, C = src.shape
        if self.training:
            T = self.num_frames
        else:
            T = window_tokens.shape[1]
        B = BT // T

        # object decoder
        dec_src = src.view(N, B, T, C).permute(2, 0, 1, 3).flatten(0,1) #[T*N, B, C]
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)     # N, B, C
        tgt = torch.zeros_like(query_embed)

        clip_hs = self.clip_decoder(tgt, dec_src, query_pos=query_embed, is_train=is_train) #[3, 100, 8, 256]

        return clip_hs


class IFCEncoder(nn.Module):

    def __init__(self, num_frames, num_queries, window_size,
                    encoder_layer, num_layers, use_checkpoint=False):
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.num_layers = num_layers
        self.enc_layers = _get_clones(encoder_layer, num_layers)
        self.use_checkpoint = use_checkpoint

    def forward(self, output):
        
        # calculate attention mask for SW-MSA
        if self.training:
            T = self.num_frames
        else:
            T = output.shape[1]
        N = self.num_queries
        Tp = int(np.ceil(T / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Tp, N, 1), device=output.device)  # 1 T N 1 
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        #给每个window clip编号
        cnt = 0
        for w in w_slices:
            img_mask[:, w, :, :] = cnt
            cnt += 1
        #mask window也进window切分部分，执行roll等操作。
        mask_windows = window_partition(img_mask, self.window_size)  # nW*1, window_size, N, 1
        mask_windows = mask_windows.view(-1, self.window_size * N)
        #通过广播操作相减，使得index相同的部分attn_mask上为0，后续根据attn_mask上的值处理
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW*1, window_size*N, window_size*N
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # #定义一个不可学习的变量，attn_mask为属性名。
        # self.register_buffer("attn_mask", attn_mask)

        for layer_idx in range(self.num_layers):
            blocks= self.enc_layers[layer_idx]  # output:[100,1*6,256][N, B*T, C]
            for blk in blocks:
                if self.use_checkpoint:
                    output = checkpoint.checkpoint(blk, output, attn_mask) #https://www.cnblogs.com/jiangkejie/p/13049684.html
                else:
                    output = blk(output, attn_mask)

        return output


def window_partition(x, window_size):
    B, T, N, C = x.shape  # [B, T, N, C]
    x = x.view(B, T // window_size, window_size, N, C) #[1, 19, 100, 256]-->
    windows = x.view(-1, window_size, N, C) # nW*B, window_size, N, C
    return windows

def window_reverse(windows, window_size, T, N):
    B = int(windows.shape[0] / (T / window_size)) # nW*B, window_size, N, C
    x = windows.view(B, T // window_size, window_size, N, -1)
    x = x.view(B, T, N, -1) # [B, T, N, C]
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim_attenion, num_queries, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim_attenion = dim_attenion
        self.window_size = window_size  # Ww,
        self.num_heads = num_heads
        head_dim = dim_attenion // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_queries = num_queries

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.num_queries - 1) * (2 * window_size - 1), self.num_heads))  # 2*N-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.num_queries)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, N, T
        coords_flatten = torch.flatten(coords, 1)  # 2, N*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N*Ww, N*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N*Ww, N*Ww, 2
        relative_coords[:, :, 0] += self.num_queries - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # N*Ww, N*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim_attenion, dim_attenion * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attenion, dim_attenion)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N_, C = x.shape  # nW*B, window_size*N, C
        qkv = self.qkv(x).reshape(B_, N_, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_queries * self.window_size, self.num_queries * self.window_size, -1)  # window_size*N,window_size*N,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, window_size*N, window_size*N
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None: # nW, window_size*N, window_size*N
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_, N_) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_, N_)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim_attenion={self.dim_attenion}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    def __init__(self, batch_size, num_queries, dim_attenion, num_heads, window_size=6, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.batch_size = batch_size
        self.dim_attenion = dim_attenion # C的通道数 默认256
        self.num_queries = num_queries   # 默认为100个object_queries
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim_attenion)
        self.attn = WindowAttention(
            dim_attenion, num_queries = self.num_queries, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_attenion)
        mlp_hidden_dim = int(dim_attenion * mlp_ratio)
        self.mlp = Mlp(in_features=dim_attenion, hidden_features=mlp_hidden_dim, act_layer=act_layer, dropout=dropout)

    def forward(self, x, mask_matrix):
        if self.training:
            B = self.batch_size
        else:
            B = 1
        N, L, C = x.shape # [N, B*T, C] 推理时为[100, 36, 256]
        T = L//B

        x = self.norm1(x)
        x = x.view(N, B, T, C).permute(1, 2, 0, 3).contiguous()  #[B, T, N, C]
        shortcut = x.view(B, T*N, C) #[B, T*N, C]

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - T % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, 0, 0, pad_r), mode='replicate') #[1, 19, 100, 256] # https://blog.csdn.net/binbinczsohu/article/details/106359426
        _, Tp, N, _ = x.shape

        # cyclic shift
        if self.shift_size > 0: #
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, N, C
        x_windows = x_windows.view(-1, self.window_size * N, C)  # nW*B, window_size*N, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*N, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, N, C) # nW*B, window_size, N, C
        shifted_x = window_reverse(attn_windows, self.window_size, Tp, C)  # [B, T, N, C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
        
        if pad_r > 0:
            x = x[:, :T, :, :].contiguous()

        x = x.view(B, T * N, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x = x.view(B, T, N, C).permute(2, 0, 1, 3).contiguous().view(N, B*T, C)  # [B, T, N, C] --> [N, B*T, C]
        return x


class IFCDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
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
        output = tgt  #相当于 Object queries [N,B,C]
        return_intermediate = (self.return_intermediate and is_train)
        intermediate = []
        
        for layer in self.layers: #dec_src:[T*N, B, C]
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


class TransformerDecoderLayer(nn.Module):

    def __init__(self, dim_attenion, num_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_attenion, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_attenion, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_attenion, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_attenion)

        self.norm1 = nn.LayerNorm(dim_attenion)
        self.norm2 = nn.LayerNorm(dim_attenion)
        self.norm3 = nn.LayerNorm(dim_attenion)
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
        q = k = self.with_pos_embed(tgt, query_pos) #video queries
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt) #[N,T,C]
        # memeory is object Tokens (也就是 K,V) [T*N, B, C]
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos), #这里detr的k使用的是绝对位置编码，后要消融实验看
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
