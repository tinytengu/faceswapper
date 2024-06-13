import pathlib

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer


class RealESRGANUpsampler:
    def __init__(self, model_path: pathlib.Path):
        self.model_path = model_path

        half = True if torch.cuda.is_available() else False
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path=str(self.model_path),
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=half,
        )
        self.instance = upsampler
