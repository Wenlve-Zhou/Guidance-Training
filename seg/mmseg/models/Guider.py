from mmseg.ops import resize
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed,Block
import numpy as np
from functools import partial
from timm.models.layers import trunc_normal_

def calibrated_pseudo_weight(pseudo_weight, mask):
    for i in range(mask.size(0)):
        ratio = torch.nonzero(mask[i]).size(0) / mask[i].numel()
        ratio = 1 - np.exp(-ratio * 5)
        pseudo_weight[i] = ratio * pseudo_weight[i]
    return pseudo_weight


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class Guider(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        print(cfg)
        self.decoder_type = cfg["single_scale_head"] if "single_scale_head" in cfg.keys() else cfg["type"]
        self.in_channels = cfg["in_channels"]
        self.in_index = cfg["in_index"]
        if self.decoder_type == "DAFormerHead":
            AssertionError("DAFormer-based Guidance Training is not yet complete, please select the DeepLab model.")

        elif self.decoder_type == "DLV2Head":
            in_spatial = 64
            spatial_dim = 64
            channel_dim = self.in_channels
            sr_ratio = 4

        self.attn = Aggregation(spatial_dim=spatial_dim, channel_dim=channel_dim, sr_ratio=sr_ratio)
        self.projector = Projector(channel_dim=self.in_channels, patch_size=sr_ratio * int(in_spatial//spatial_dim))


    def forward(self, inputs, mask):
        attn_out = self.attn(inputs[-1], mask)
        out = self.projector(attn_out, inputs[-1], mask)
        outputs = [out] * 4
        return outputs

class Projector(nn.Module):
    def __init__(self, channel_dim, patch_size, embed_dim=512, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, channel_dim, 1, 1))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * channel_dim)
        self.patch_size = patch_size
        self.channel_dim = channel_dim
        self.apply(_init_weights)

        self.z2 = nn.Conv2d(channel_dim, channel_dim, 1, 1)
        nn.init.constant_(self.z2.weight, 0.0)
        nn.init.constant_(self.z2.bias, 0.0)

    def forward(self, attn, src, mask):
        offset = self.predict_offset(attn)
        mask = resize(
            input=mask.float(),
            size=src.shape[2:],
            mode='nearest')
        x_ini = src * mask + self.mask_token * (1-mask)
        x = x_ini + offset
        return x

    def predict_offset(self, x):
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = self.unpatchify(x)
        x = self.z2(x)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * channel_dim)
        imgs: (N, channel_dim, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.channel_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        features = x.reshape(shape=(x.shape[0], self.channel_dim, h * p, h * p))
        return features

class Aggregation(nn.Module):
    def __init__(self, spatial_dim, channel_dim, sr_ratio=1, embed_dim=512, num_heads=16, mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depth=2):
        super().__init__()
        # Learnable Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, channel_dim, 1, 1))
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Patch Embedding & Blocks Construction
        self.patch_embed = PatchEmbed(img_size=spatial_dim, patch_size=sr_ratio, in_chans=channel_dim, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.apply(_init_weights)

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim),requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero Convolution Z1
        self.z1 = nn.Conv2d(channel_dim, channel_dim, 1)
        nn.init.constant_(self.z1.weight, 0.0)
        nn.init.constant_(self.z1.bias, 0.0)

    def forward(self, x, mask):
        # Remove the Source Feature in Mixture and Add Learnable Token in the Position
        B, C, H, W = x.size()
        mask = resize(
            input=mask.float(),
            size=x.shape[2:],
            mode='nearest')
        mask_token = self.mask_token.repeat(B, 1, H, W)
        x = x * mask + (1 - mask) * mask_token

        # Contextual Aggregation using Attention Mechanism
        x = self.z1(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
