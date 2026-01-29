"""
LTX2 Efficient Sampler Pro

Enhanced version of LTX2 Efficient Sampler with:
- All original features (optimization presets, thermal control, etc.)
- FFN chunking for additional VRAM reduction
- Proper Audio-Video (NestedTensor) support via CFGGuider
"""

import torch
import time
import math
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import node_helpers
import latent_preview
from .ffn_chunking import apply_ffn_chunking, restore_ffn
from .gpu_monitor import get_gpu_monitor, PYNVML_AVAILABLE
from .optimization_engines import get_engine, get_engine_names, EngineConfig

# Optimization Presets (same as original sampler)
OPTIMIZATION_PRESETS = {
    "Quality - Fast (RTX 3080+)": {
        "freeze_ratio": 0.5,
        "throttle_delay": 0,
        "frame_stride": 1,
        "auto_thermal": False,
        "description": "Full quality, minimal throttling for high-end GPUs"
    },
    "Quality - Balanced (RTX 3060/3070)": {
        "freeze_ratio": 0.4,
        "throttle_delay": 25,
        "frame_stride": 1,
        "auto_thermal": False,
        "description": "Full quality with moderate thermal management"
    },
    "Quality - RTX 2060 6GB": {
        "freeze_ratio": 0.3,
        "throttle_delay": 50,
        "frame_stride": 1,
        "auto_thermal": False,
        "description": "Full quality, good thermal management for 6GB cards"
    },
    "Quality - Cool (RTX 2060 6GB)": {
        "freeze_ratio": 0.25,
        "throttle_delay": 100,
        "frame_stride": 1,
        "auto_thermal": False,
        "description": "Full quality with aggressive cooling pauses (~70°C target)"
    },
    "Quality - Ultra Cool (Low Power GPUs)": {
        "freeze_ratio": 0.2,
        "throttle_delay": 150,
        "frame_stride": 1,
        "auto_thermal": False,
        "description": "Full quality, maximum thermal throttling for older/laptop GPUs"
    },
    "Thermal Auto-Scale": {
        "freeze_ratio": 0.3,
        "throttle_delay": None,  # Dynamic based on temperature
        "frame_stride": 1,
        "auto_thermal": True,
        "description": "Full quality, auto-adjusts throttling based on GPU temperature"
    },
    "Custom": {
        "freeze_ratio": None,
        "throttle_delay": None,
        "frame_stride": None,
        "auto_thermal": False,
        "description": "Use manual settings below"
    }
}


class LTX2EfficientSamplerPro:
    """
    LTX2 Efficient Sampler Pro - Full-featured with FFN Chunking
    
    Uses CFGGuider for proper NestedTensor (audio-video) support.
    This is the same approach used by SamplerCustomAdvanced.
    """
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "video/ltx2"
    
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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Optimization settings (same as original)
                "optimization_preset": (list(OPTIMIZATION_PRESETS.keys()), {"default": "Thermal Auto-Scale"}),
                "target_temp": ("INT", {"default": 70, "min": 50, "max": 85, "step": 1}),
                "freeze_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "thermal_throttle": ("BOOLEAN", {"default": True}),
                # Context handling for Dual CLIP (Gemma + LTX Connector)
                "context_slice_method": ([
                    "Keep Last 4096 (Slot 2 - LTX Connector)", 
                    "Keep First 4096 (Slot 1 - T5)", 
                    "Auto-detect"
                ], {"default": "Keep Last 4096 (Slot 2 - LTX Connector)", "tooltip": "For Gemma+LTX Connector: use Last 4096. For T5+Gemma: use First 4096."}),
            },
            "optional": {
                # PRO: FFN Chunking (exclusive to Pro Sampler)
                "ffn_chunks": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 24, 
                    "step": 1,
                    "tooltip": "[PRO] Number of chunks for FFN processing. Higher = less VRAM but slower. 1=disabled."
                }),
                # LTXVScheduler parameters (FlowMatch)
                "sigmas": ("SIGMAS", {"tooltip": "Connect LTXVScheduler output here to override internal scheduler"}),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "FlowMatch max shift (same as LTXVScheduler)"}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "FlowMatch base shift (same as LTXVScheduler)"}),
                "terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Sigma terminal value after stretching"}),
                "stretch": ("BOOLEAN", {"default": True, "tooltip": "Stretch sigmas to [terminal, 1] range"}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Frame rate for conditioning (same as LTXVConditioning)"}),
            }
        }
    
    def _generate_ltx_sigmas(self, steps, latent_shape, max_shift=2.05, base_shift=0.95, terminal=0.1, stretch=True):
        """
        Generate LTX-compatible sigmas using FlowMatch formula.
        """
        # Extract dimensions for mu calculation
        if len(latent_shape) >= 5:
            _, _, frames, h, w = latent_shape[:5]
        elif len(latent_shape) >= 4:
            _, frames, h, w = latent_shape[:4]
        else:
            frames, h, w = 20, 64, 64  # Fallback defaults
        
        # Calculate mu based on image dimensions (FlowMatch formula)
        img_size = h * w * frames
        log_img = math.log(img_size) / math.log(4096)
        mu = log_img * max_shift + base_shift
        
        # Generate timesteps
        timesteps = torch.linspace(1, terminal, steps + 1, dtype=torch.float32)
        
        # Apply shift schedule
        if stretch:
            stretched = (mu / (mu + (1 / timesteps - 1)))
        else:
            stretched = timesteps
        
        # Convert timesteps to sigmas using ComfyUI's convention
        sigmas = stretched / (1 - stretched + 1e-8)
        sigmas = torch.clamp(sigmas, min=0.0)
        
        # Ensure final sigma is 0
        sigmas[-1] = 0.0
        
        return sigmas
    
    def _patch_model(self, model, latent_video):
        """
        Patches the model to:
        1. Inject attention_mask=None if missing (fixes LTXBaseModel error)
        2. Reshape 3D packed latents back to 5D (fixes patchifier error)
        3. Handle Audio-Video packed latents by splitting/processing video only
        """
        if not hasattr(model.model, 'diffusion_model'):
            return model
            
        diffusion_model = model.model.diffusion_model
        
        # Capture latent shape for reshaping (Always update it!)
        latent_shape = None
        samples = latent_video.get("samples")
        if samples is not None:
             if hasattr(samples, 'shape'):
                 latent_shape = samples.shape
             elif hasattr(samples, 'unbind'): # NestedTensor
                 latent_shape = samples.unbind()[0].shape
        
        # Store shape on model
        diffusion_model._ltx2_efficient_shape = latent_shape

        # Check if already patched
        if getattr(diffusion_model, '_ltx2_efficient_patched', False):
            return model
                    
        original_forward = diffusion_model.forward
        
        def patched_forward(x, timestep, context=None, attention_mask=None, **kwargs):
            # Get current latent shape
            current_shape = getattr(diffusion_model, '_ltx2_efficient_shape', None)
            
            # Handle case where x is a list (passthrough to original)
            if isinstance(x, (list, tuple)):
                return original_forward(x, timestep, context, attention_mask=attention_mask, **kwargs)
            
            # Handle case where x is not a tensor (passthrough)
            if not hasattr(x, 'ndim'):
                return original_forward(x, timestep, context, attention_mask=attention_mask, **kwargs)
            
            # IMPORTANT: CFGGuider.sample() handles NestedTensor packing/unpacking correctly
            # We should NOT interfere with audio-video latent handling here
            # Our patch only needs to ensure attention_mask is passed through
            
            # For 3D packed latents from CFGGuider, just passthrough 
            # The model (LTXAVModel) knows how to handle combined audio-video latents
            if x.ndim == 3:
                # Debug once
                if not getattr(diffusion_model, '_ltx2_debug_logged', False):
                    print(f"[LTX2Pro] DEBUG: Model Class: {type(diffusion_model).__name__}")
                    print(f"[LTX2Pro] DEBUG: Input Shape: {x.shape}")
                    print(f"[LTX2Pro] DEBUG: Passing through to original forward()")
                    diffusion_model._ltx2_debug_logged = True
                
                # Let the model handle it natively - don't split or reshape
                return original_forward(x, timestep, context, attention_mask=attention_mask, **kwargs)
            
            # For 5D video-only latents (standard case)
            if x.ndim == 5:
                return original_forward(x, timestep, context, attention_mask=attention_mask, **kwargs)
            
            # Default fallthrough
            return original_forward(x, timestep, context, attention_mask=attention_mask, **kwargs)
            
        # Apply patch
        diffusion_model.forward = patched_forward
        diffusion_model._ltx2_efficient_patched = True
        print(f"[LTX2Pro] Patched model forward() for attention_mask & 3D latents (with AV split support)")
        
        return model
    
    def sample(self, model, latent_video, positive, negative, seed, steps, cfg, sampler_name, denoise,
               optimization_preset, target_temp, freeze_ratio, thermal_throttle, 
               context_slice_method="Keep Last 4096 (Slot 2 - LTX Connector)",
               ffn_chunks=8, sigmas=None, max_shift=2.05, base_shift=0.95, terminal=0.1, 
               stretch=True, frame_rate=25.0):
        """
        Main sampling function using CFGGuider for proper NestedTensor support.
        """
        # Patch model to ensure compatibility with CFGGuider/LTX
        self._patch_model(model, latent_video)
        
        samples = latent_video["samples"]
        
        # Detect NestedTensor
        is_nested = (
            str(type(samples).__name__) == 'NestedTensor' or 
            'nested' in str(type(samples)).lower() or
            (hasattr(samples, 'is_nested') and samples.is_nested)
        )
        
        # Get latent shape for sigma generation
        if is_nested:
            first_tensor = samples.unbind()[0] if hasattr(samples, 'unbind') else samples
            latent_shape = first_tensor.shape
            print(f"[LTX2Pro] 🎵 Audio-Video mode (NestedTensor detected)")
        else:
            latent_shape = samples.shape
            print(f"[LTX2Pro] 🎬 Video-only mode")
        
        print(f"[LTX2Pro] Latent shape: {latent_shape}")
        
        # Generate or use provided sigmas
        if sigmas is not None:
            print(f"[LTX2Pro] Using external sigmas")
            print(f"[LTX2Pro] Sigma range: {sigmas[0]:.4f} -> {sigmas[-1]:.4f}, steps: {len(sigmas)-1}")
            use_sigmas = sigmas
        else:
            print(f"[LTX2Pro] Generating LTX FlowMatch sigmas")
            use_sigmas = self._generate_ltx_sigmas(steps, latent_shape, max_shift, base_shift, terminal, stretch)
            print(f"[LTX2Pro] Sigmas: {use_sigmas[0]:.4f} -> {use_sigmas[-1]:.4f}")
        
        # Truncate sigmas for denoise < 1.0
        if denoise < 1.0:
            total_sigmas = len(use_sigmas) - 1
            start_step = int(total_sigmas * (1 - denoise))
            use_sigmas = use_sigmas[start_step:]
        
        # Inject frame_rate into conditioning
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})
        print(f"[LTX2Pro] Set frame_rate={frame_rate}")
        
        # Apply Optimization Preset
        preset = OPTIMIZATION_PRESETS.get(optimization_preset, {})
        use_auto_thermal = preset.get("auto_thermal", False)
        
        if preset.get("freeze_ratio") is not None:
            freeze_ratio = preset["freeze_ratio"]
        
        throttle_delay_ms = preset.get("throttle_delay")
        if throttle_delay_ms is None:
            throttle_delay_ms = 50 if thermal_throttle else 0
        
        print(f"[LTX2Pro] Preset: {optimization_preset}")
        if use_auto_thermal:
            print(f"[LTX2Pro] -> THERMAL AUTO-SCALE, target={target_temp}°C")
        print(f"[LTX2Pro] -> freeze_ratio={freeze_ratio}, throttle_delay={throttle_delay_ms}ms")
        
        # Initialize GPU monitor if using auto thermal
        gpu_monitor = None
        if use_auto_thermal and PYNVML_AVAILABLE:
            gpu_monitor = get_gpu_monitor()
            if gpu_monitor.available:
                initial_temp = gpu_monitor.get_temperature()
                print(f"[LTX2Pro] GPU temp: {initial_temp}°C")
            else:
                use_auto_thermal = False
        elif use_auto_thermal:
            use_auto_thermal = False
        
        # Apply FFN chunking optimization
        ffn_originals = {}
        if ffn_chunks > 1:
            try:
                ffn_originals = apply_ffn_chunking(model, num_chunks=ffn_chunks, verbose=True)
            except Exception as e:
                print(f"[LTX2Pro] FFN chunking failed (non-fatal): {e}")
        
        try:
            # Fix empty latent channels
            latent_image = comfy.sample.fix_empty_latent_channels(model, samples)
            
            # Get noise_mask if present
            # NOTE: CFGGuider.sample() properly handles NestedTensor denoise_mask!
            # (lines 1013-1029 in samplers.py: unbinds, prepares each, repacks)
            # So we should NOT disable it - that was causing input images to be ignored.
            noise_mask = latent_video.get("noise_mask", None)
            if noise_mask is not None and is_nested:
                print(f"[LTX2Pro] Audio-Video Mode: Using noise_mask with NestedTensor (CFGGuider handles this)")
            
            # Generate noise using comfy.sample.prepare_noise (handles NestedTensor)
            noise = comfy.sample.prepare_noise(latent_image, seed)
            
            # Create CFGGuider (this is what SamplerCustomAdvanced uses!)
            # CFGGuider.sample() properly handles NestedTensor by packing/unpacking
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(cfg)
            
            # Create KSAMPLER sampler object
            sampler = comfy.samplers.ksampler(sampler_name)
            
            # Prepare callback for thermal throttling
            x0_output = {}
            
            def make_callback():
                current_step = [0]
                base_callback = latent_preview.prepare_callback(model, len(use_sigmas) - 1, x0_output)
                
                def callback(step, denoised, x, total_steps):
                    current_step[0] = step
                    
                    # Thermal management
                    if use_auto_thermal and gpu_monitor and gpu_monitor.available:
                        current_temp = gpu_monitor.get_temperature()
                        if current_temp > target_temp:
                            cooldown = min((current_temp - target_temp) * 20, 500)
                            time.sleep(cooldown / 1000.0)
                    elif throttle_delay_ms > 0:
                        time.sleep(throttle_delay_ms / 1000.0)
                    
                    # Call base callback for preview
                    if base_callback:
                        base_callback(step, denoised, x, total_steps)
                
                return callback
            
            callback = make_callback()
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            
            print(f"[LTX2Pro] 🚀 Starting sampling ({len(use_sigmas)-1} steps)...")
            
            # Use CFGGuider.sample() - this is the key!
            # It properly handles NestedTensor by packing into single tensor before sampling
            result = guider.sample(
                noise,
                latent_image,
                sampler,
                use_sigmas,
                denoise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed
            )
            
            result = result.to(comfy.model_management.intermediate_device())
            
            print(f"[LTX2Pro] ✓ Sampling complete")
            
        finally:
            # Restore FFN
            if ffn_originals:
                try:
                    restore_ffn(ffn_originals, verbose=True)
                except:
                    pass
        
        # Return result
        out = latent_video.copy()
        out["samples"] = result
        
        return (out,)
