import torch
import torch.nn as nn
from torch.nn import functional as F

from vgg_frontend import build_vgg
from linear import LinearModel
from dil_conv_block import build_dc
from utils import initialize_weights
from transformer import Transformer


class RelScaleTransPos(nn.Module):
    def __init__(self, backbone, transformer, linear, pos_embed_x, pos_embed_z) -> None:
        super(RelScaleTransPos, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.head = linear
        self.pos_embed_x = pos_embed_x
        self.pos_embed_z = pos_embed_z
        self.bottleneck = nn.Conv2d(backbone.num_channels, 256, kernel_size=1)
        initialize_weights(self.transformer)
        initialize_weights(self.head)

    def forward_backbone(self, img: torch.Tensor, zx: str):
        assert isinstance(img, torch.Tensor)
        output_back = self.backbone(img)
        bs = img.size(0)  #batch size
        if zx == 'search':
            pos = self.pos_embed_x(bs)
        elif zx == 'map':
            pos = self.pos_embed_z(bs)
        else:
            raise ValueError('zx must be search or map')
        
        mask = torch.ones(size=(bs, img.shape[-1], img.shape[-1]))
        mask_down = F.interpolate(mask[None].float(), size=output_back.shape[-2:]).to(torch.bool)[0]
        return self.adjust(output_back, pos, mask_down)

    def get_qkv(inp_list):
        """The 1st element of the inp_list is about the template,
        the 2nd (the last) element is about the search region"""
        dict_x = inp_list[-1]
        dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
                "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
                "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
        q = dict_x["feat"] + dict_x["pos"]
        k = dict_c["feat"] + dict_c["pos"]
        v = dict_c["feat"]
        key_padding_mask = dict_c["mask"]
        return q, k, v, key_padding_mask


    def forward_transformer(self, q, k, v, key_padding_mask=None, softmax=True):
        enc_mem = self.transformer(q, k, v, key_padding_mask=key_padding_mask)
        out = self.head(enc_mem)
        return out


    def forward(self, img1, img2):
        img1 = img1.to(torch.float32)
        img2 = img2.to(torch.float32)

        x1 = self.forward_backbone(img1, zx='search')
        x2 = self.forward_backbone(img2, zx='map')
       
        # x1 = torch.flatten(self.dc1(x1), start_dim=1)
        # x2 = torch.flatten(self.dc2(x2), start_dim=1)
       
        input_catted = torch.cat((torch.flatten(x1), torch.flatten(x2)), dim=1)

        input_catted = input_catted.view(-1, 1, 1024)
        print(input_catted.shape)
        #ste grum enq forward transformery
        out = self.transformer(input_catted, input_catted)
        out = out.view(-1, 8192)
        print(out.shape)

        out = torch.flatten(out, start_dim=1)
        out = self.lin(out)
        return out   

    def adjust(self, src_feat: torch.Tensor, pos_embed: torch.Tensor, mask: torch.Tensor):
        """
        """
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

# def build_relscaletransformer(cfg, pretrained):
#     model = RelScaleTransformer()
#     if pretrained:
#         checkpoint = torch.load('./models/sh_bal_data_model_6_epoch.pth')
#         model.load_state_dict(checkpoint['model_state_dict'])
#     return model