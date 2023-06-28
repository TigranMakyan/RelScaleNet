import torch
import os

from stark_scale import build_stark_scale_model
from repvgg import repvgg_model_convert
from image_utils import PreprocessorX


class STARK_LightningXtrt:
    def __init__(self, params, dataset_name):
        super(STARK_LightningXtrt, self).__init__(params)
        network = build_stark_scale_model(params.cfg, phase='test')
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        repvgg_model_convert(network)
        network.deep_sup = False  # disable deep supervision during the test stage
        network.distill = False  # disable distillation during the test stage
        self.cfg = params.cfg
        self.network = network
        self.network.eval()
        self.preprocessor = PreprocessorX()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.z_dict1 = {}

    def forward(image1, image2):
        pass