"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from typing import Optional
from utils_model import NestedTensor

class PositionEmbeddingLearned_new(nn.Module):
    """
    Absolute pos embedding, learned. (allow users to specify the size)
    """
    def __init__(self, num_pos_feats=256, sz=20):
        super().__init__()
        self.sz = sz
        self.row_embed = nn.Embedding(sz, num_pos_feats)
        self.col_embed = nn.Embedding(sz, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, bs):
        """bs: batch size"""
        h, w = self.sz, self.sz
        i = torch.arange(w, device=self.col_embed.weight.device)
        j = torch.arange(h, device=self.row_embed.weight.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1)
        return pos  # (H,W,C) --> (C,H,W) --> (1,C,H,W) --> (B,C,H,W)


def build_position_encoding_new(cfg, sz):
    N_steps = cfg.MODEL.HIDDEN_DIM // 2
    position_embedding = PositionEmbeddingLearned_new(N_steps, sz)
    return position_embedding



# from backbone import build_backbone_x
# from utils_model import read_config, params_count_lite

# cfg_path = '/home/user/computer_vision/stark_scale/configs/stark.yaml'
# config = read_config(cfg_path)
# backbone = build_backbone_x(config)

# pos_emb_x = build_position_encoding_new(config, 40)

# a = pos_emb_x(1)
# print(a.shape)
# pos_emb_z = build_position_encoding_new(config, 64)
# img = torch.randn(1, 3, 512, 512)
# zx = 'search'

# import torch.nn.functional as F


# def adjust(src_feat: torch.Tensor, pos_embed: torch.Tensor, mask: torch.Tensor):
#         """
#         """
#         bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
#         # reduce channel
#         feat = bottleneck(src_feat)  # (B, C, H, W)
#         # adjust shapes
#         feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
#         pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
#         mask_vec = mask.flatten(1)  # BxHW
#         return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}






# output_back = backbone(img)  
# bs = img.size(0)
# pos = pos_emb_x(bs)
# mask_down = F.interpolate(mask[None].float(), size=output_back.shape[-2:]).to(torch.bool)[0]
# # features & masks, position embedding for the search
# """get the positional encoding"""
# bs = img.size(0)  # batch size
# if zx == "search":
#     pos = pos_emb_x(bs)
# elif "template" in zx:
#     pos = pos_emb_z(bs)
# else:
#     raise ValueError("zx should be 'template_0' or 'search'.")
# """get the downsampled attention mask"""
# mask_down = F.interpolate(mask[None].float(), size=output_back.shape[-2:]).to(torch.bool)[0]
# """adjust the shape"""
# return adjust(output_back, pos, mask_down)
