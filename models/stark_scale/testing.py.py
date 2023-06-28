# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# import yaml
# from easydict import EasyDict
# import torch.nn.functional as F

# from image_utils import PreprocessorX
# from backbone import build_backbone_x
# from pos_embed import build_position_encoding_new
# from transformer_lite import build_lite_encoder
# from utils_model import read_config, get_qkv

# process = PreprocessorX()

# img1 = cv2.imread('/home/user/Documents/arshak/image.jpg')
# img2 = cv2.imread('/home/user/Documents/arshak/image.jpg')

# img1_tensor, mask1_tensor = process.process(img1, 320)
# img2_tensor, mask2_tensor = process.process(img2, 320)

# # img1_tensor = torch.unsqueeze(img1_tensor, dim=0)
# # img2_tensor = torch.unsqueeze(img2_tensor, dim=0)

# cfg = read_config('/home/user/computer_vision/stark_scale/configs/scale.yaml')

# backbone = build_backbone_x(cfg)
# # bottleneck = nn.Conv2d(1024, 128, kernel_size=1)
# bottleneck = nn.Sequential(
#     nn.Conv2d(1024, 128, kernel_size=1),
#     nn.AvgPool2d(kernel_size=2)
# )
# fsz_x, fsz_z = cfg.DATA.SEARCH.FEAT_SIZE, cfg.DATA.TEMPLATE.FEAT_SIZE
# pos_enc_x = build_position_encoding_new(cfg, fsz_x)
# pos_enc_z = build_position_encoding_new(cfg, fsz_z)

# transformer = build_lite_encoder(cfg)
# print(img1_tensor.shape)
# print(mask1_tensor.shape)

# a = backbone(img1_tensor)
# b = backbone(img2_tensor)
# print(a.shape)
# bs = 1

# pos_x = pos_enc_x(bs)
# pos_z = pos_enc_z(bs)
# print(f'pos_x shape: {pos_x.shape}')

# a_mask_down = F.interpolate(mask1_tensor[None].float(), size=[20, 20]).to(torch.bool)[0]
# b_mask_down = F.interpolate(mask2_tensor[None].float(), size=[20, 20]).to(torch.bool)[0]

# print('a mask down', a_mask_down.shape)

# def adjust(src_feat: torch.Tensor, pos_embed: torch.Tensor, mask: torch.Tensor, bottleneck):
#         """
#         """
#         # reduce channel
#         feat = bottleneck(src_feat)  # (B, C, H, W)
#         print(f'after bottleneck: {feat.shape}')
#         # adjust shapes
#         feat_vec = feat.flatten(2).permute(2, 0, 1)
#         print(f'feat vec in adjust : {feat_vec.shape}')  # HWxBxC
#         pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
#         mask_vec = mask.flatten(1)  # BxHW
#         return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

# a_out = adjust(a, pos_x, a_mask_down, bottleneck)
# b_out = adjust(b, pos_z, b_mask_down, bottleneck)
# print(f'before transformer: {a_out["feat"].shape} and {a_out["pos"].shape}')
# input_transformer = [a_out, b_out]


# '''

# MINCHEV STEX ASHKIS TOSHNIA ETUM

# '''
# q, k, v, key_padding_mask = get_qkv(input_transformer)
# print('*'*75)
# print(q.shape)
# print(k.shape)
# print(v.shape)
# print(key_padding_mask.shape)

# memory = transformer(q, k, v, key_padding_mask)
# print(memory.shape)

# '''
# after transformer shape is 1600, bs, 256
# '''
# from head import build_head

# head = build_head(cfg=cfg)
# print(head)
# memory = torch.permute(memory, (1, 0, 2))
# print(memory.shape)
# res = head(memory)
# print(res.shape)




