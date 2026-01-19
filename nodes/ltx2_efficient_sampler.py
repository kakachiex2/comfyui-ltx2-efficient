import torch
import comfy.samplers
import comfy.sample
import nodes
from .interpolation import interpolate
from .gpu_monitor import get_gpu_monitor, PYNVML_AVAILABLE

# Optimization Presets for different GPU tiers
OPTIMIZATION_PRESETS = {
    "Performance (RTX 3080+)": {
        "freeze_ratio": 0.7,
        "throttle_delay": 0,
        "frame_stride": 2,
        "auto_thermal": False,
        "description": "Minimal throttling for high-end GPUs"
    },
    "Balanced (RTX 3060/3070)": {
        "freeze_ratio": 0.5,
        "throttle_delay": 25,
        "frame_stride": 4,
        "auto_thermal": False,
        "description": "Good balance of speed and temperature"
    },
    "Aggressive (RTX 2060/2070)": {
        "freeze_ratio": 0.3,
        "throttle_delay": 50,
        "frame_stride": 4,
        "auto_thermal": False,
        "description": "Current optimization - good for 6GB cards"
    },
    "Ultra Low Power (GTX 1660/Laptop)": {
        "freeze_ratio": 0.15,
        "throttle_delay": 100,
        "frame_stride": 6,
        "auto_thermal": False,
        "description": "Maximum throttling for older/mobile GPUs"
    },
    "Extreme Cool (RTX 2060 6GB)": {
        "freeze_ratio": 0.1,
        "throttle_delay": 150,
        "frame_stride": 8,
        "auto_thermal": False,
        "description": "Target 65°C - very aggressive throttling"
    },
    "Thermal Auto-Scale": {
        "freeze_ratio": 0.3,
        "throttle_delay": None,  # Dynamic based on temperature
        "frame_stride": 4,
        "auto_thermal": True,
        "description": "Reads GPU temp and auto-adjusts throttling"
    },
    "Custom": {
        "freeze_ratio": None,
        "throttle_delay": None,
        "frame_stride": None,
        "auto_thermal": False,
        "description": "Use manual settings below"
    }
}

class LTX2EfficientSampler:
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
                "optimization_preset": (list(OPTIMIZATION_PRESETS.keys()), {"default": "Thermal Auto-Scale"}),
                "target_temp": ("INT", {"default": 70, "min": 50, "max": 85, "step": 1}),
                "frame_stride": ("INT", {"default": 4, "min": 1, "max": 32}),
                "attention_window": ("INT", {"default": 4, "min": 1, "max": 32}),
                "freeze_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "thermal_throttle": ("BOOLEAN", {"default": True}),
                "interpolation_method": (["linear", "slerp", "motion"], {"default": "slerp"}),
                "audio_passthrough": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "video/ltx2"
    
    @classmethod
    def IS_CHANGED(cls, model, latent_video, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, optimization_preset, target_temp, frame_stride, attention_window, freeze_ratio, thermal_throttle, interpolation_method, audio_passthrough):
        """
        ComfyUI caching mechanism.
        Returns a hash of inputs that affect the output.
        If this returns the same value as the previous run, cached output is used.
        """
        import hashlib
        
        # Create a fingerprint of the key parameters that affect output
        # Note: We include seed so different seeds produce different results
        # We exclude target_temp and thermal_throttle as they don't affect output, only speed
        fingerprint = f"{seed}_{steps}_{cfg}_{sampler_name}_{scheduler}_{denoise}_{frame_stride}_{freeze_ratio}_{interpolation_method}_{audio_passthrough}"
        
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def sample(self, model, latent_video, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, optimization_preset, target_temp, frame_stride, attention_window, freeze_ratio, thermal_throttle, interpolation_method, audio_passthrough):
        import time
        
        # Audio passthrough: detect and preserve audio latent if present
        audio_latent = None
        if audio_passthrough:
            # Check if latent_video contains audio (tuple format from LTXVImgToVideo)
            if isinstance(latent_video.get("samples"), tuple):
                video_samples, audio_samples = latent_video["samples"]
                audio_latent = {"samples": audio_samples}
                latent_video = {"samples": video_samples}
                print(f"[LTX2Efficient] Audio passthrough enabled. Audio latent preserved.")
            else:
                print(f"[LTX2Efficient] Audio passthrough enabled but no audio found in input.")
        
        # Apply Optimization Preset (override manual settings unless "Custom")
        preset = OPTIMIZATION_PRESETS.get(optimization_preset, {})
        use_auto_thermal = preset.get("auto_thermal", False)
        
        if preset.get("freeze_ratio") is not None:
            freeze_ratio = preset["freeze_ratio"]
        if preset.get("frame_stride") is not None:
            frame_stride = preset["frame_stride"]
        
        # Get throttle delay, defaulting to 50ms if None (for auto-thermal fallback)
        throttle_delay_ms = preset.get("throttle_delay")
        if throttle_delay_ms is None:
            throttle_delay_ms = 50 if thermal_throttle else 0
        
        print(f"[LTX2Efficient] Using preset: {optimization_preset}")
        if use_auto_thermal:
            print(f"[LTX2Efficient] -> THERMAL AUTO-SCALE enabled, target_temp={target_temp}°C")
        print(f"[LTX2Efficient] -> freeze_ratio={freeze_ratio}, throttle_delay={throttle_delay_ms}ms, frame_stride={frame_stride}")
        
        # Initialize GPU monitor if using auto thermal
        gpu_monitor = None
        if use_auto_thermal and PYNVML_AVAILABLE:
            gpu_monitor = get_gpu_monitor()
            if gpu_monitor.available:
                initial_temp = gpu_monitor.get_temperature()
                print(f"[LTX2Efficient] GPU Monitor active. Current temp: {initial_temp}°C")
            else:
                print(f"[LTX2Efficient] Warning: GPU Monitor unavailable, falling back to fixed throttle.")
                use_auto_thermal = False
        elif use_auto_thermal and not PYNVML_AVAILABLE:
            print(f"[LTX2Efficient] Warning: pynvml not installed, falling back to fixed throttle.")
            use_auto_thermal = False
        
        # 1. Keyframe Selection
        samples = latent_video["samples"]
        # Assume (Frames, Channels, Height, Width)
        original_frames_count = samples.shape[0]
        
        # Select indices
        indices = torch.arange(0, original_frames_count, frame_stride)
        keyframes = samples[indices]
        
        kf_latent = {"samples": keyframes}
        
        # 2. Setup Patching
        # We create a patcher that attempts to intercept attention calls.
        # Note: This is a simplified "best effort" patcher for the LTX2 architecture context.
        
        # We define a context manager or update mechanism.
        # Since standard KSampler doesn't easily allow per-step callbacks to modify model *topology*, 
        # we rely on global/shared state or modifying the model options.
        
        # Ideally, we would clone the model and wrap its diffusion_model.
        # But for now, we will perform the sampling with the provided model,
        # hoping the model supports the standard ComfyUI attention or we can patch it globally temporarily.
        
        # WARNING: Global patching is risky. We should try to use model_options if possible.
        # But attention windowing usually requires low-level injection.
        
        # Let's try to pass the windowing parameters via model_options and hope the underlying model (if custom) respects them,
        # OR we try to find the attention modules and patch them.
        
        # Strategy: Recursive module search and patch.
        patcher = LTX2Patcher(model, attention_window)
        patcher.apply()
        
        try:
            # CALLBACK for freezing and thermal management
            def step_callback(step, x0, x, total_steps):
                t = step / total_steps
                
                # THERMAL THROTTLE (DYNAMIC or FIXED)
                if use_auto_thermal and gpu_monitor is not None:
                    # Dynamic: Read current temp and calculate delay
                    current_temp = gpu_monitor.get_temperature()
                    dynamic_delay = gpu_monitor.calculate_throttle_delay(
                        current_temp, target_temp, min_delay=0, max_delay=300
                    )
                    
                    if step % 5 == 0:  # Log every 5 steps
                        print(f"[LTX2Efficient] Step {step}: Temp={current_temp}°C, Delay={dynamic_delay}ms")
                    
                    # Emergency pause if critical
                    if gpu_monitor.should_emergency_pause(current_temp, critical_temp=83):
                        print(f"[LTX2Efficient] ⚠️ CRITICAL TEMP {current_temp}°C! Emergency 2s pause.")
                        time.sleep(2.0)
                    elif dynamic_delay > 0:
                        time.sleep(dynamic_delay / 1000.0)
                    
                    torch.cuda.empty_cache()
                    
                elif throttle_delay_ms > 0:
                    # Fixed delay from preset
                    time.sleep(throttle_delay_ms / 1000.0)
                    torch.cuda.empty_cache()
                
                # Check patcher state
                if t > freeze_ratio:
                    if not patcher.spatial_frozen:
                        print(f"[LTX2Efficient] Freezing spatial at step {step} ({t:.1%})")
                        patcher.freeze_spatial()
                        # Optional: aggressive cleanup when switching modes
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    patcher.unfreeze_spatial()
            
            # 3. Run Sampler
            # nodes.common_ksampler doesn't support callback, so we use comfy.sample.sample directly
            
            # Unwrap latent dictionary to get the tensor
            kf_samples = kf_latent["samples"]
            current_frames = kf_samples.shape[0]
            print(f"[LTX2Efficient] Sampling on reduced frames: {kf_samples.shape} (Stride={frame_stride})")
            
            # Update patcher with frame info for heuristics
            patcher.set_frame_info(current_frames)
            
            # Prepare noise
            # comfy.sample.prepare_noise expects a Tensor (latent_image), not a dict
            noise = comfy.sample.prepare_noise(kf_samples, seed)
            
            # Run sampling
            # comfy.sample.sample also expects the latent Tensor as 'latent_image'
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
            
            # comfy.sample.sample returns a Tensor
            # We wrap it back into a dictionary for the rest of our logic
            result = {"samples": result_tensor}
            
        finally:
            patcher.restore()

        # 4. Interpolate
        result_samples = result["samples"]
        print(f"[LTX2Efficient] Interpolating with method: {interpolation_method}")
        full_samples = interpolate(result_samples, frame_stride, method=interpolation_method)
        
        # Handle potential length mismatch due to stride not dividing perfectly
        if full_frames_count := full_samples.shape[0]:
             if full_frames_count > original_frames_count:
                 full_samples = full_samples[:original_frames_count]
        
        # Return with or without audio
        if audio_passthrough and audio_latent is not None:
            # Return combined format for LTXVSeparateAVLatent compatibility
            combined_samples = (full_samples, audio_latent["samples"])
            print(f"[LTX2Efficient] Returning combined video+audio latent.")
            return ({"samples": combined_samples},)
        else:
            return ({"samples": full_samples},)

class LTX2Patcher:
    """
    LTX2-Specific Attention Patcher
    
    Based on LTX-Video architecture (LTXVideoTransformerBlock):
    - attn1: Self-attention (spatiotemporal self-attention)
    - attn2: Cross-attention (for conditioning like text prompts)
    
    Spatial attention freezing skips attn1 after freeze_ratio.
    Cross-attention (attn2) is always preserved for prompt adherence.
    """
    
    # LTX2 specific module patterns
    LTX2_ATTENTION_PATTERNS = [
        "attn1",  # Self-attention (spatial-temporal)
        "attn2",  # Cross-attention (text conditioning)
    ]
    
    def __init__(self, model, window, debug=False):
        self.model = model
        self.window = window
        self.original_forward_maps = {}
        self.spatial_frozen = False
        self.patched_count = 0
        self.attn1_count = 0
        self.attn2_count = 0
        self.current_frames_count = 1
        self.debug = debug
        self.module_names = {}  # Store module path -> type mapping
        
    def set_frame_info(self, frames):
        self.current_frames_count = frames
        
    def apply(self):
        # Locate the underlying diffusion model
        if hasattr(self.model, "model"):
             model_obj = self.model.model
        else:
             model_obj = self.model
             
        if hasattr(model_obj, "diffusion_model"):
            diffusion_model = model_obj.diffusion_model
        else:
            print(f"[LTX2Efficient] Warning: Could not find 'diffusion_model' in {type(model_obj)}.")
            return

        self.patched_count = 0
        self.attn1_count = 0
        self.attn2_count = 0
        
        # LTX2-specific patching: look for attn1/attn2 modules
        # IMPORTANT: Only patch the Attention module itself, NOT its sublayers (to_q, to_k, etc.)
        for name, module in diffusion_model.named_modules():
            # Must be an actual Attention class (not Linear, LayerNorm, etc.)
            class_name = module.__class__.__name__
            if not class_name.endswith("Attention"):
                continue
            
            # Check if this is an LTX2 attention module by path
            # Path should END with attn1 or attn2 (not contain deeper sublayers)
            is_attn1 = name.endswith(".attn1") or name.endswith("attn1")
            is_attn2 = name.endswith(".attn2") or name.endswith("attn2")
            
            if is_attn1 or is_attn2:
                attn_type = "attn1" if is_attn1 else "attn2"
            else:
                attn_type = "generic"
            
            if self._patch_module(module, attn_type, name):
                self.patched_count += 1
                if is_attn1:
                    self.attn1_count += 1
                elif is_attn2:
                    self.attn2_count += 1
                
                if self.debug:
                    print(f"[LTX2Efficient] Patched [{attn_type}]: {name}")
        
        print(f"[LTX2Efficient] Patched {self.patched_count} attention layers (attn1={self.attn1_count}, attn2={self.attn2_count})")

    def _patch_module(self, module, attn_type, module_path):
        if module in self.original_forward_maps:
            return False
            
        original_forward = module.forward
        self.original_forward_maps[module] = original_forward
        self.module_names[module] = (attn_type, module_path)
        
        def new_forward(x, context=None, *args, **kwargs):
            # === LTX2-SPECIFIC OPTIMIZATION ===
            
            # attn1 = Self-attention (spatiotemporal) - CAN BE FROZEN
            # attn2 = Cross-attention (text conditioning) - ALWAYS RUN
            
            if attn_type == "attn1":
                # Self-attention: Apply spatial freeze optimization
                if self.spatial_frozen:
                    # Skip computation, return zeros (residual will preserve signal)
                    if self.debug:
                        print(f"[LTX2Efficient] FROZEN: {module_path}")
                    return torch.zeros_like(x)
            
            elif attn_type == "attn2":
                # Cross-attention: Always execute for prompt adherence
                pass
            
            else:
                # Generic attention: Use shape heuristic
                B = x.shape[0]
                is_spatial = False
                
                if self.current_frames_count > 0 and B % self.current_frames_count == 0:
                    ratio = B // self.current_frames_count
                    if ratio <= 4:
                        is_spatial = True
                
                if is_spatial and self.spatial_frozen:
                    return torch.zeros_like(x)
            
            # Normal execution
            return original_forward(x, context, *args, **kwargs)
            
        module.forward = new_forward 
        return True

    def freeze_spatial(self):
        self.spatial_frozen = True
        
    def unfreeze_spatial(self):
        self.spatial_frozen = False
        
    def restore(self):
        for module, original in self.original_forward_maps.items():
            module.forward = original
        self.original_forward_maps.clear()
        self.module_names.clear()
