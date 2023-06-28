import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_frontend import build_vgg

bb = build_vgg(pretrained=True, freeze=True)

def adjust(self, src_feat: torch.Tensor, pos_embed: torch.Tensor, mask: torch.Tensor):
        """
        """
        bottlenecknn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # reduce channel
        feat = bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

def forward_backbone(bb, pos_embed_x, pos_embed_z, img: torch.Tensor, zx: str, mask: torch.Tensor):
    assert isinstance(img, torch.Tensor)

    output_back = bb(img)

    bs = img.size(0)

    if zx == 'search':
        pos = pos_embed_x(output_back)
    elif zx == 'map':
        pos = pos_embed_z(output_back)
    else:
        raise ValueError('zx should be map or search')

    mask_down = F.interpolate(mask[None].float(), size=output_back.shape[-2:]).to(torch.bool)[0]
    return adjust(output_back, pos, mask_down)