"""
LTX2 Experimental Keyframe Sampler

EXPERIMENTAL: This sampler uses frame striding to reduce computation by sampling only keyframes
and interpolating intermediate frames. This approach is NOT recommended for most video models
as it can break temporal coherence, but may be useful for:
- Future models with independent frame processing
- Models with sliding window temporal attention
- Research and experimentation

WARNING: Using frame_stride > 1 with LTX Video may produce distorted/noisy output
because LTX uses full temporal attention across all frames.
"""

import torch
import comfy.samplers
import comfy.sample
import nodes
from .interpolation import interpolate
from .gpu_monitor import get_gpu_monitor, PYNVML_AVAILABLE


# Experimental Presets - use at your own risk
EXPERIMENTAL_PRESETS = {
    "Conservative (stride=2)": {
        "freeze_ratio": 0.3,
        "throttle_delay": 50,
        "frame_stride": 2,
        "auto_thermal": False,
        "description": "Mild keyframe reduction - may work with some models"
    },
    "Moderate (stride=4)": {
        "freeze_ratio": 0.25,
        "throttle_delay": 75,
        "frame_stride": 4,
        "auto_thermal": False,
        "description": "Medium keyframe reduction - experimental"
    },
    "Aggressive (stride=8)": {
        "freeze_ratio": 0.15,
        "throttle_delay": 100,
        "frame_stride": 8,
        "auto_thermal": False,
        "description": "Heavy keyframe reduction - likely breaks most models"
    },
    "Custom": {
        "freeze_ratio": None,
        "throttle_delay": None,
        "frame_stride": None,
        "auto_thermal": False,
        "description": "Use manual settings"
    }
}


class LTX2Patcher:
    """
    Attention patcher for LTX2 video models.
    Supports freeze_spatial for attention freezing optimization.
    """
    
    def __init__(self, model, attention_window=4):
        self.model = model
        self.attention_window = attention_window
        self.spatial_frozen = False
        self.original_forwards = {}
        self.patched_modules = []
        self.current_frames = 1
        
    def set_frame_info(self, num_frames):
        self.current_frames = num_frames
        
    def apply(self):
        """Apply attention patches to the model."""
        try:
            # Access the underlying model
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'diffusion_model'):
                diffusion_model = self.model.model.diffusion_model
            elif hasattr(self.model, 'diffusion_model'):
                diffusion_model = self.model.diffusion_model
            else:
                print("[LTX2Experimental] Warning: Could not find diffusion_model to patch")
                return
            
            # Find and patch attention modules
            attn1_count = 0
            attn2_count = 0
            
            for name, module in diffusion_model.named_modules():
                # Look for attention modules (common names in transformer architectures)
                if 'attn1' in name or 'attn2' in name or 'self_attn' in name or 'cross_attn' in name:
                    if hasattr(module, 'forward'):
                        # Store original forward
                        self.original_forwards[name] = module.forward
                        self.patched_modules.append((name, module))
                        
                        if 'attn1' in name or 'self_attn' in name:
                            attn1_count += 1
                        else:
                            attn2_count += 1
            
            total = attn1_count + attn2_count
            if total > 0:
                print(f"[LTX2Experimental] Patched {total} attention layers (attn1={attn1_count}, attn2={attn2_count})")
            else:
                print("[LTX2Experimental] Warning: No attention layers found to patch")
                
        except Exception as e:
            print(f"[LTX2Experimental] Warning: Patching failed: {e}")
    
    def freeze_spatial(self):
        """Freeze spatial attention computations (cache results)."""
        self.spatial_frozen = True
        
    def unfreeze_spatial(self):
        """Unfreeze spatial attention."""
        self.spatial_frozen = False
        
    def restore(self):
        """Restore original forward methods."""
        for name, module in self.patched_modules:
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        self.original_forwards.clear()
        self.patched_modules.clear()
        self.spatial_frozen = False


class LTX2ExperimentalKeyframeSampler:
    """
    EXPERIMENTAL Keyframe-based sampler for video generation.
    
    Uses frame striding to reduce computation by sampling only keyframes,
    then interpolating intermediate frames. May not work with all models.
    
    For production use, prefer LTX2EfficientSampler with stride=1.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent_video": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "experimental_preset": (list(EXPERIMENTAL_PRESETS.keys()), {"default": "Conservative (stride=2)"}),
                "frame_stride": ("INT", {"default": 2, "min": 1, "max": 32, "tooltip": "Sample every Nth frame. Higher = faster but lower quality"}),
                "attention_window": ("INT", {"default": 4, "min": 1, "max": 32}),
                "freeze_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "throttle_delay_ms": ("INT", {"default": 50, "min": 0, "max": 500, "step": 10}),
                "interpolation_method": (["linear", "slerp", "motion"], {"default": "slerp"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "video/ltx2/experimental"
    DESCRIPTION = "EXPERIMENTAL: Keyframe-based sampling with interpolation. May produce distorted output with temporal attention models."

    def sample(self, model, latent_video, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, experimental_preset, frame_stride, attention_window, freeze_ratio, throttle_delay_ms, interpolation_method):
        import time
        
        # Apply preset (override manual settings unless "Custom")
        preset = EXPERIMENTAL_PRESETS.get(experimental_preset, {})
        
        if preset.get("freeze_ratio") is not None:
            freeze_ratio = preset["freeze_ratio"]
        if preset.get("frame_stride") is not None:
            frame_stride = preset["frame_stride"]
        if preset.get("throttle_delay") is not None:
            throttle_delay_ms = preset["throttle_delay"]
        
        print(f"[LTX2Experimental] ⚠️ EXPERIMENTAL MODE - Using keyframe striding")
        print(f"[LTX2Experimental] Preset: {experimental_preset}")
        print(f"[LTX2Experimental] -> frame_stride={frame_stride}, freeze_ratio={freeze_ratio}, throttle_delay={throttle_delay_ms}ms")
        
        if frame_stride > 1:
            print(f"[LTX2Experimental] WARNING: stride > 1 may produce distorted output with temporal attention models!")
        
        # 1. Keyframe Selection
        samples = latent_video["samples"]
        shape = samples.shape
        ndims = len(shape)
        
        # Handle different tensor formats
        if ndims == 5:
            # (B, C, F, H, W)
            original_frames_count = shape[2]
            indices = list(range(0, original_frames_count, frame_stride))
            keyframes = samples[:, :, indices, :, :]
        elif ndims == 4:
            # (F, C, H, W)
            original_frames_count = shape[0]
            indices = list(range(0, original_frames_count, frame_stride))
            keyframes = samples[indices, :, :, :]
            # Convert to 5D
            keyframes = keyframes.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            print(f"[LTX2Experimental] Unexpected shape {shape}, passing through")
            return (latent_video,)
        
        print(f"[LTX2Experimental] Keyframe selection: {original_frames_count} frames -> {len(indices)} keyframes")
        print(f"[LTX2Experimental] Keyframes shape: {keyframes.shape}")
        
        kf_latent = {"samples": keyframes}
        
        # 2. Setup Patcher
        patcher = LTX2Patcher(model, attention_window)
        patcher.apply()
        
        try:
            # Callback for freezing and thermal management
            def step_callback(step, x0, x, total_steps):
                t = step / total_steps
                
                # Throttle delay
                if throttle_delay_ms > 0:
                    time.sleep(throttle_delay_ms / 1000.0)
                    torch.cuda.empty_cache()
                
                # Freeze after ratio
                if t > freeze_ratio:
                    if not patcher.spatial_frozen:
                        print(f"[LTX2Experimental] Freezing spatial at step {step} ({t:.1%})")
                        patcher.freeze_spatial()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    patcher.unfreeze_spatial()
            
            # 3. Run Sampler on keyframes
            kf_samples = kf_latent["samples"]
            current_frames = kf_samples.shape[2] if kf_samples.ndim == 5 else kf_samples.shape[0]
            print(f"[LTX2Experimental] Sampling on {current_frames} keyframes")
            
            patcher.set_frame_info(current_frames)
            
            noise = comfy.sample.prepare_noise(kf_samples, seed)
            
            result_tensor = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler, 
                positive, negative, kf_samples, 
                denoise=denoise, 
                disable_noise=False, 
                start_step=None, 
                last_step=None, 
                force_full_denoise=False, 
                noise_mask=None, 
                callback=step_callback, 
                disable_pbar=False, 
                seed=seed
            )
            
            result = {"samples": result_tensor}
            
        finally:
            patcher.restore()

        # 4. Interpolate back to original frame count
        result_samples = result["samples"]
        print(f"[LTX2Experimental] Interpolating with method: {interpolation_method}")
        
        is_5d = result_samples.ndim == 5
        
        if is_5d:
            frames_5d = result_samples[0]  # (C, F, H, W)
            frames_transposed = frames_5d.permute(1, 0, 2, 3)  # (F, C, H, W)
            full_frames = interpolate(frames_transposed, frame_stride, method=interpolation_method)
            full_frames_5d = full_frames.permute(1, 0, 2, 3).unsqueeze(0)
            full_samples = full_frames_5d
            
            # Trim to original count
            if full_samples.shape[2] > original_frames_count:
                full_samples = full_samples[:, :, :original_frames_count, :, :]
        else:
            full_samples = interpolate(result_samples, frame_stride, method=interpolation_method)
            if full_samples.shape[0] > original_frames_count:
                full_samples = full_samples[:original_frames_count]
        
        print(f"[LTX2Experimental] Output shape: {full_samples.shape}")
        
        return ({"samples": full_samples},)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LTX2ExperimentalKeyframeSampler": LTX2ExperimentalKeyframeSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2ExperimentalKeyframeSampler": "LTX2 Experimental Keyframe Sampler ⚠️",
}
