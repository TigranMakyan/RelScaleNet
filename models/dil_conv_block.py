import torch
import torch.nn as nn

from models.utils import initialize_weights

def make_layers(cfg, in_channels = 3,batch_norm=True, dilation = True, d=2):
    '''
    example of config => [512, 512, 512, 256, 256, 128 ,64, 64, 32, 16]
    '''
    if dilation:
        d_rate = d
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Tanh()]
            else:
                layers += [conv2d, nn.Tanh()]
            in_channels = v
    return nn.Sequential(*layers)


class Dil_Conv_Block(nn.Module):
    def __init__(self, cfg, d_rate=2):
        super(Dil_Conv_Block, self).__init__()
        self.cfg  = cfg
        self.dc = make_layers(self.cfg, in_channels = 512,dilation = True, d=d_rate)
        
    
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.dc(x)
        return x



def build_dc(freeze=False):
    model = Dil_Conv_Block(cfg=[512, 512, 512, 256, 256, 128 ,64, 64, 32, 16])
    initialize_weights(model)
    if freeze:
        print('[INFO]: FREEZE Dilated Conv block')
        for params in model.parameters():
            params.requires_grad = False
    return model
    




# model =build_dc()
# print(model)
# a = torch.rand(1, 512, 16, 16) #==> 1, 4096
# b = torch.rand(1, 512, 8, 8) #==> 1, 1024
# res = model(b)
# print(res.shape)
