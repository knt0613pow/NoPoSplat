from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg, DecoderSplattingCUDA2D

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
    "splatting_cuda_2d" : DecoderSplattingCUDA2D,
}

DecoderCfg = DecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)

def get_3d_decoder(decoder_cfg: DecoderCfg) -> DecoderSplattingCUDA:
    return DECODERS["splatting_cuda"](decoder_cfg)

def get_2d_decoder(decoder_cfg: DecoderCfg) -> DecoderSplattingCUDA2D:
    return DECODERS["splatting_cuda_2d"](decoder_cfg)