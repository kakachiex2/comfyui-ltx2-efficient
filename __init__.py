from .nodes.ltx2_efficient_sampler import LTX2EfficientSampler
from .nodes.ltx2_vae_decode import LTX2TemporalVAEDecode
from .nodes.ltx2_latent_utils import (
    LTX2SeparateAVLatent,
    LTX2CombineAVLatent,
    LTX2EmptyAudioLatent
)

NODE_CLASS_MAPPINGS = {
    "LTX2EfficientSampler": LTX2EfficientSampler,
    "LTX2TemporalVAEDecode": LTX2TemporalVAEDecode,
    "LTX2SeparateAVLatent": LTX2SeparateAVLatent,
    "LTX2CombineAVLatent": LTX2CombineAVLatent,
    "LTX2EmptyAudioLatent": LTX2EmptyAudioLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2EfficientSampler": "LTX2 Efficient Video Sampler",
    "LTX2TemporalVAEDecode": "LTX2 Temporal VAE Decode",
    "LTX2SeparateAVLatent": "LTX2 Separate Audio/Video Latent",
    "LTX2CombineAVLatent": "LTX2 Combine Audio/Video Latent",
    "LTX2EmptyAudioLatent": "LTX2 Empty Audio Latent",
}
