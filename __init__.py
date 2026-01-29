from .nodes.ltx2_efficient_sampler import LTX2EfficientSampler
from .nodes.ltx2_efficient_sampler_pro import LTX2EfficientSamplerPro
from .nodes.ltx2_vae_decode import LTX2TemporalVAEDecode
from .nodes.ltx2_latent_utils import (
    LTX2SeparateAVLatent,
    LTX2CombineAVLatent,
    LTX2EmptyAudioLatent
)
from .nodes.ltx2_experimental_sampler import LTX2ExperimentalKeyframeSampler
from .nodes.ltx2_conditioning import LTX2ConditioningHelper
from .nodes.ltx2_model_patcher import LTX2ModelPatcherNode
from .nodes.ltx2_text_encode_optimized import LTX2TextEncodeOptimized

NODE_CLASS_MAPPINGS = {
    "LTX2EfficientSampler": LTX2EfficientSampler,
    "LTX2EfficientSamplerPro": LTX2EfficientSamplerPro,
    "LTX2TemporalVAEDecode": LTX2TemporalVAEDecode,
    "LTX2SeparateAVLatent": LTX2SeparateAVLatent,
    "LTX2CombineAVLatent": LTX2CombineAVLatent,
    "LTX2EmptyAudioLatent": LTX2EmptyAudioLatent,
    "LTX2ExperimentalKeyframeSampler": LTX2ExperimentalKeyframeSampler,
    "LTX2ConditioningHelper": LTX2ConditioningHelper,
    "LTX2ModelPatcher": LTX2ModelPatcherNode,
    "LTX2TextEncodeOptimized": LTX2TextEncodeOptimized,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2EfficientSampler": "LTX2 Efficient Video Sampler",
    "LTX2EfficientSamplerPro": "LTX2 Efficient Sampler Pro 🚀",
    "LTX2TemporalVAEDecode": "LTX2 Temporal VAE Decode",
    "LTX2SeparateAVLatent": "LTX2 Separate Audio/Video Latent",
    "LTX2CombineAVLatent": "LTX2 Combine Audio/Video Latent",
    "LTX2EmptyAudioLatent": "LTX2 Empty Audio Latent",
    "LTX2ExperimentalKeyframeSampler": "LTX2 Experimental Keyframe Sampler ⚠️",
    "LTX2ConditioningHelper": "LTX2 Conditioning Helper",
    "LTX2ModelPatcher": "LTX2 Model Patcher",
    "LTX2TextEncodeOptimized": "LTX2 Text Encode (Optimized) 💾",
}

