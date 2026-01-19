from .nodes.ltx2_efficient_sampler import LTX2EfficientSampler
from .nodes.ltx2_vae_decode import LTX2TemporalVAEDecode

NODE_CLASS_MAPPINGS = {
    "LTX2EfficientSampler": LTX2EfficientSampler,
    "LTX2TemporalVAEDecode": LTX2TemporalVAEDecode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2EfficientSampler": "LTX2 Efficient Video Sampler",
    "LTX2TemporalVAEDecode": "LTX2 Temporal VAE Decode"
}
