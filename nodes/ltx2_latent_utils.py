"""
LTX2 Efficient Latent Utilities
Audio-Video Latent Separation and Combination nodes for efficient processing.
"""

import torch


class AVLatentWrapper:
    """
    A wrapper class that mimics NestedTensor's API for audio-video latent compatibility.
    
    This allows LTX2CombineAVLatent output to be compatible with:
    - LTXVSeparateAVLatent (expects .unbind() to return [video, audio])
    - LTXVDecodeAV (expects NestedTensor-like behavior)
    - Other LTX nodes that check for NestedTensor
    
    Methods implemented:
    - unbind(): Returns [video, audio] tensors
    - values(): Returns video tensor (primary content)
    - to_padded_tensor(padding): Returns video tensor (for compatibility)
    - clone(): Creates a copy
    - contiguous(): Returns self (tensors are already contiguous)
    - is_contiguous(): Returns True
    
    Properties:
    - shape: Returns video tensor shape
    - device: Returns video tensor device
    - dtype: Returns video tensor dtype
    - ndim: Returns video tensor ndim
    """
    
    def __init__(self, video_tensor, audio_tensor):
        self.video = video_tensor
        self.audio = audio_tensor
        # Store shape info for compatibility checks
        self._shape = video_tensor.shape
    
    def unbind(self):
        """Return video and audio tensors like NestedTensor.unbind()"""
        return [self.video, self.audio]
    
    def values(self):
        """Return the primary tensor (video) for NestedTensor compatibility"""
        return self.video
    
    def to_padded_tensor(self, padding=0.0):
        """Return video tensor for NestedTensor compatibility"""
        return self.video
    
    def clone(self):
        """Create a copy of this wrapper"""
        return AVLatentWrapper(self.video.clone(), self.audio.clone())
    
    def contiguous(self):
        """Return self with contiguous tensors"""
        return AVLatentWrapper(self.video.contiguous(), self.audio.contiguous())
    
    def is_contiguous(self):
        """Check if underlying tensors are contiguous"""
        return self.video.is_contiguous() and self.audio.is_contiguous()
    
    def to(self, *args, **kwargs):
        """Move tensors to device/dtype"""
        return AVLatentWrapper(
            self.video.to(*args, **kwargs),
            self.audio.to(*args, **kwargs)
        )
    
    def detach(self):
        """Detach tensors from computation graph"""
        return AVLatentWrapper(self.video.detach(), self.audio.detach())
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def device(self):
        return self.video.device
    
    @property
    def dtype(self):
        return self.video.dtype
    
    @property
    def ndim(self):
        return self.video.ndim
    
    def __repr__(self):
        return f"AVLatentWrapper(video={self.video.shape}, audio={self.audio.shape})"
    
    def __len__(self):
        """Return 2 (video and audio)"""
        return 2

class LTX2SeparateAVLatent:
    """
    Separates combined audio-video latent into individual video and audio latents.
    
    Use this to extract video latent for efficient sampling, while preserving
    audio latent for recombination later.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "create_empty_audio": ("BOOLEAN", {"default": True, "tooltip": "Create empty audio latent if none found"}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "BOOLEAN")
    RETURN_NAMES = ("video_latent", "audio_latent", "has_audio")
    FUNCTION = "separate"
    CATEGORY = "video/ltx2"
    DESCRIPTION = "Separates combined audio-video latent for efficient processing. Route video to LTX2EfficientSampler."

    def _is_nested_tensor(self, tensor):
        """Check if tensor is a NestedTensor or AVLatentWrapper."""
        type_name = str(type(tensor).__name__)
        type_str = str(type(tensor)).lower()
        # Recognize both NestedTensor and our custom AVLatentWrapper
        return type_name == 'NestedTensor' or 'nested' in type_str or type_name == 'AVLatentWrapper'

    def _extract_from_nested(self, nested_tensor):
        """
        Try to extract video and audio tensors from a NestedTensor.
        Returns (video_tensor, audio_tensor) or (None, None) if extraction fails.
        """
        # Method 1: Try unbind - NestedTensor with 2 components
        if hasattr(nested_tensor, 'unbind'):
            try:
                tensors = nested_tensor.unbind()
                if len(tensors) == 2:
                    # Assume first is video (larger), second is audio
                    t0, t1 = tensors
                    # Video is 5D (B,C,F,H,W), Audio is 4D (B,C,F,AudioChannels)
                    if len(t0.shape) == 5 and len(t1.shape) == 4:
                        return t0, t1
                    elif len(t1.shape) == 5 and len(t0.shape) == 4:
                        return t1, t0
                    # Both same dimensionality - assume first is video
                    return t0, t1
            except Exception as e:
                print(f"[LTX2SeparateAVLatent] unbind failed: {e}")
        
        # Method 2: Try to_padded_tensor - returns single tensor, can't separate
        # This method doesn't help for separation
        
        # Method 3: Try accessing internal _values
        if hasattr(nested_tensor, '_values'):
            try:
                values = nested_tensor._values
                if isinstance(values, (list, tuple)) and len(values) == 2:
                    return values[0], values[1]
            except Exception as e:
                print(f"[LTX2SeparateAVLatent] _values access failed: {e}")
        
        return None, None

    def _create_empty_audio_latent(self, video_samples):
        """
        Create an empty audio latent matching the video latent's frame count.
        
        LTX Audio latent format: (B, C, AudioFrames, AudioChannels)
        - B: batch size (from video)
        - C: latent channels (typically 64 for audio VAE)
        - AudioFrames: based on video frames and sample rate
        - AudioChannels: typically 1 (mono in latent space)
        """
        # Get video dimensions
        if len(video_samples.shape) == 5:
            batch, _, video_frames, _, _ = video_samples.shape
        elif len(video_samples.shape) == 4:
            video_frames, _, _, _ = video_samples.shape
            batch = 1
        else:
            batch = 1
            video_frames = 1
        
        # Audio latent dimensions
        # LTX uses 16kHz audio, 25 FPS video
        # Audio latent frames ≈ video_frames * (16000/160/4) / 25 ≈ video_frames
        audio_channels = 64  # Standard audio VAE latent channels
        audio_frames = video_frames  # Approximate matching
        audio_width = 1  # Mono
        
        # Create zeros tensor
        empty_audio = torch.zeros(
            batch, audio_channels, audio_frames, audio_width,
            dtype=video_samples.dtype,
            device=video_samples.device
        )
        
        return empty_audio

    def separate(self, latent, create_empty_audio=True):
        samples = latent.get("samples")
        
        print(f"[LTX2SeparateAVLatent] Input type: {type(samples).__name__}, shape: {getattr(samples, 'shape', 'N/A')}")
        
        # Case 1: Tuple format (video, audio) - from some LTX nodes
        if isinstance(samples, tuple) and len(samples) == 2:
            video_samples, audio_samples = samples
            print(f"[LTX2SeparateAVLatent] Found tuple format. Video: {video_samples.shape}, Audio: {audio_samples.shape}")
            return (
                {"samples": video_samples},
                {"samples": audio_samples},
                True
            )
        
        # Case 2: NestedTensor - from LTXVConcatAVLatent
        if self._is_nested_tensor(samples):
            print(f"[LTX2SeparateAVLatent] Detected NestedTensor, attempting extraction...")
            video_tensor, audio_tensor = self._extract_from_nested(samples)
            
            if video_tensor is not None and audio_tensor is not None:
                print(f"[LTX2SeparateAVLatent] Successfully extracted. Video: {video_tensor.shape}, Audio: {audio_tensor.shape}")
                return (
                    {"samples": video_tensor},
                    {"samples": audio_tensor},
                    True
                )
            else:
                print(f"[LTX2SeparateAVLatent] Could not extract from NestedTensor. Treating as video-only.")
                # Try to convert NestedTensor to regular tensor for video
                try:
                    if hasattr(samples, 'to_padded_tensor'):
                        video_tensor = samples.to_padded_tensor(0.0)
                    elif hasattr(samples, 'unbind'):
                        video_tensor = samples.unbind()[0]
                    else:
                        video_tensor = samples
                except:
                    video_tensor = samples
                samples = video_tensor
        
        # Case 3: Regular tensor - video only
        if isinstance(samples, torch.Tensor):
            print(f"[LTX2SeparateAVLatent] Regular tensor: {samples.shape}. Video only.")
            
            if create_empty_audio:
                empty_audio = self._create_empty_audio_latent(samples)
                print(f"[LTX2SeparateAVLatent] Created empty audio latent: {empty_audio.shape}")
                return (
                    {"samples": samples},
                    {"samples": empty_audio},
                    False
                )
            else:
                return (
                    {"samples": samples},
                    None,
                    False
                )
        
        # Fallback - pass through unchanged
        print(f"[LTX2SeparateAVLatent] Warning: Unrecognized format, passing through.")
        if create_empty_audio:
            # Try to create empty audio anyway
            try:
                empty_audio = torch.zeros(1, 64, 1, 1, dtype=torch.float32)
                return (latent, {"samples": empty_audio}, False)
            except:
                pass
        return (latent, None, False)


class LTX2CombineAVLatent:
    """
    Combines video and audio latents for downstream LTX nodes.
    
    Use after efficient sampling to recombine processed video with audio.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
            },
            "optional": {
                "audio_latent": ("LATENT",),
                # Default to nested_tensor for compatibility with LTXVSeparateAVLatent and LTXVDecodeAV
                "output_format": (["nested_tensor", "tuple", "video_only"], {"default": "nested_tensor"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("combined_latent",)
    FUNCTION = "combine"
    CATEGORY = "video/ltx2"
    DESCRIPTION = "Combines video and audio latents for downstream LTX decode nodes. Use 'nested_tensor' for LTXVSeparateAVLatent/LTXVDecodeAV compatibility."

    def combine(self, video_latent, audio_latent=None, output_format="nested_tensor"):
        video_samples = video_latent["samples"]
        
        # Video only - no audio
        if audio_latent is None or output_format == "video_only":
            print(f"[LTX2CombineAVLatent] Video only output: {video_samples.shape}")
            return ({"samples": video_samples},)
        
        audio_samples = audio_latent["samples"]
        print(f"[LTX2CombineAVLatent] Combining Video: {video_samples.shape}, Audio: {audio_samples.shape}")
        
        if output_format == "nested_tensor":
            # Use AVLatentWrapper - compatible with LTXVSeparateAVLatent and LTXVDecodeAV
            # The standard LTX nodes expect .unbind() to return [video, audio] tensors
            # Note: We use our wrapper because torch.nested.nested_tensor requires
            # same-dimension tensors, but video is 5D and audio is 4D
            combined = AVLatentWrapper(video_samples, audio_samples)
            print(f"[LTX2CombineAVLatent] Created AVLatentWrapper with Video: {video_samples.shape}, Audio: {audio_samples.shape}")
            return ({"samples": combined},)
        
        elif output_format == "tuple":
            # Simple tuple format - for nodes that accept tuple input
            combined = (video_samples, audio_samples)
            return ({"samples": combined},)
        
        # Default fallback
        return ({"samples": video_samples},)


class LTX2EmptyAudioLatent:
    """
    Creates an empty audio latent matching a video latent's timing.
    
    Use when you need a placeholder audio latent for nodes that require both
    video and audio inputs, but you only have video.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
            },
            "optional": {
                "audio_channels": ("INT", {"default": 64, "min": 1, "max": 256, "tooltip": "Audio VAE latent channels"}),
                "duration_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Multiplier for audio duration relative to video"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("audio_latent",)
    FUNCTION = "create_empty"
    CATEGORY = "video/ltx2"
    DESCRIPTION = "Creates empty audio latent matching video duration. Use as placeholder for audio-requiring nodes."

    def create_empty(self, video_latent, audio_channels=64, duration_multiplier=1.0):
        video_samples = video_latent["samples"]
        
        # Handle different video formats
        if isinstance(video_samples, tuple):
            video_samples = video_samples[0]
        
        # Get video frame count
        if len(video_samples.shape) == 5:
            batch, _, video_frames, _, _ = video_samples.shape
        elif len(video_samples.shape) == 4:
            video_frames = video_samples.shape[0]
            batch = 1
        else:
            video_frames = 1
            batch = 1
        
        # Calculate audio frames with multiplier
        audio_frames = max(1, int(video_frames * duration_multiplier))
        
        # Create empty audio latent
        # Format: (B, C, AudioFrames, AudioWidth)
        empty_audio = torch.zeros(
            batch, 
            audio_channels, 
            audio_frames, 
            1,  # Mono width
            dtype=video_samples.dtype,
            device=video_samples.device
        )
        
        print(f"[LTX2EmptyAudioLatent] Created empty audio: {empty_audio.shape} (video had {video_frames} frames)")
        
        return ({"samples": empty_audio},)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LTX2SeparateAVLatent": LTX2SeparateAVLatent,
    "LTX2CombineAVLatent": LTX2CombineAVLatent,
    "LTX2EmptyAudioLatent": LTX2EmptyAudioLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2SeparateAVLatent": "LTX2 Separate Audio/Video Latent",
    "LTX2CombineAVLatent": "LTX2 Combine Audio/Video Latent",
    "LTX2EmptyAudioLatent": "LTX2 Empty Audio Latent",
}
