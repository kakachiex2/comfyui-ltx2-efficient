import torch
import math
import comfy.samplers
import comfy.sample
import comfy.model_management
import node_helpers
import nodes
from .interpolation import interpolate
from .gpu_monitor import get_gpu_monitor, PYNVML_AVAILABLE
from .optimization_engines import get_engine, get_engine_names, EngineConfig

# Optimization Presets for different GPU tiers
# NOTE: frame_stride=1 is required for quality output. LTX models use temporal attention
# across all frames - skipping frames breaks coherence. Efficiency comes from freeze_ratio
# (freezes spatial attention after X% of steps) and throttle_delay (cooling pauses).
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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Optimization settings (kept in original order for backward compatibility)
                "optimization_preset": (list(OPTIMIZATION_PRESETS.keys()), {"default": "Thermal Auto-Scale"}),
                "target_temp": ("INT", {"default": 70, "min": 50, "max": 85, "step": 1}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 32, "tooltip": "Use 1 for best quality"}),
                "attention_window": ("INT", {"default": 4, "min": 1, "max": 32}),
                "freeze_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "thermal_throttle": ("BOOLEAN", {"default": True}),
                "interpolation_method": (["linear", "slerp", "motion", "none"], {"default": "slerp"}),
                # Optimization Engine settings
                "optimization_engine": (get_engine_names(), {"default": "Adaptive Cache (2-4x speedup)", "tooltip": "Temporal attention optimization engine"}),
                "engine_cache_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Fraction of later steps to cache (AdaCache)"}),
                "engine_merge_ratio": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Fraction of tokens to merge (TokenMerge)"}),
                # Context handling for Dual CLIP (Gemma + LTX Connector)
                "context_slice_method": ([
                    "Keep Last 4096 (Slot 2 - LTX Connector)", 
                    "Keep First 4096 (Slot 1 - T5)", 
                    "Auto-detect"
                ], {"default": "Keep Last 4096 (Slot 2 - LTX Connector)", "tooltip": "For Gemma+LTX Connector: use Last 4096. For T5+Gemma: use First 4096."}),
            },
            "optional": {
                # LTXVScheduler parameters (FlowMatch) - optional for backward compatibility
                "sigmas": ("SIGMAS", {"tooltip": "Connect LTXVScheduler output here to override internal scheduler"}),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "FlowMatch max shift (same as LTXVScheduler)"}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "FlowMatch base shift (same as LTXVScheduler)"}),
                "terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Sigma terminal value after stretching"}),
                "stretch": ("BOOLEAN", {"default": True, "tooltip": "Stretch sigmas to [terminal, 1] range"}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Frame rate for conditioning (same as LTXVConditioning)"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "video/ltx2"
    
    @classmethod
    def IS_CHANGED(cls, model, latent_video, positive, negative, seed, steps, cfg, sampler_name, denoise, 
                   optimization_preset, target_temp, frame_stride, attention_window, freeze_ratio, 
                   thermal_throttle, interpolation_method, optimization_engine, engine_cache_ratio, 
                   engine_merge_ratio, context_slice_method="Keep Last 4096 (Slot 2 - LTX Connector)",
                   sigmas=None, max_shift=2.05, base_shift=0.95, terminal=0.1, stretch=True, frame_rate=25.0):
        """
        ComfyUI caching mechanism.
        Returns a unique value when output would change.
        """
        import hashlib
        
        fingerprint_parts = [
            str(id(model)),
            str(id(latent_video)),
            str(id(positive)),
            str(id(negative)),
            str(seed),
            str(steps),
            str(cfg),
            str(sampler_name),
            str(denoise),
            str(max_shift),
            str(base_shift),
            str(terminal),
            str(stretch),
            str(frame_rate),
            str(optimization_preset),
            str(frame_stride),
            str(attention_window),
            str(freeze_ratio),
            str(interpolation_method),
            str(optimization_engine),
            str(engine_cache_ratio),
            str(engine_merge_ratio),
            str(context_slice_method),
            str(id(sigmas)) if sigmas is not None else "none",
        ]
        
        fingerprint = "_".join(fingerprint_parts)
        return hashlib.md5(fingerprint.encode()).hexdigest()
    
    def _generate_ltx_sigmas(self, steps, latent_shape, max_shift=2.05, base_shift=0.95, terminal=0.1, stretch=True):
        """
        Generate LTX-compatible sigmas using FlowMatch formula.
        This replicates LTXVScheduler from ComfyUI's nodes_lt.py
        """
        # Calculate token count from latent shape (B, C, F, H, W)
        if len(latent_shape) == 5:
            tokens = latent_shape[2] * latent_shape[3] * latent_shape[4]  # F * H * W
        elif len(latent_shape) == 4:
            tokens = latent_shape[0] * latent_shape[2] * latent_shape[3]  # F * H * W
        else:
            tokens = 4096  # Default fallback
        
        # Generate linear sigmas
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
        
        # Calculate sigma shift based on token count (same formula as LTXVScheduler)
        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b
        
        # Apply FlowMatch formula
        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )
        
        # Stretch sigmas so that its final value matches the given terminal value
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched
        
        return sigmas

    def sample(self, model, latent_video, positive, negative, seed, steps, cfg, sampler_name, denoise,
               optimization_preset, target_temp, frame_stride, attention_window, freeze_ratio, 
               thermal_throttle, interpolation_method, optimization_engine, engine_cache_ratio, 
               engine_merge_ratio, context_slice_method="Keep Last 4096 (Slot 2 - LTX Connector)",
               sigmas=None, max_shift=2.05, base_shift=0.95, terminal=0.1, stretch=True, frame_rate=25.0):
        import time
        
        # Generate or use provided sigmas
        latent_shape = latent_video["samples"].shape
        if sigmas is not None:
            # Use externally provided sigmas (from LTXVScheduler)
            print(f"[LTX2Efficient] Using external sigmas (e.g., from LTXVScheduler)")
            print(f"[LTX2Efficient] Sigma range: {sigmas[0]:.4f} -> {sigmas[-1]:.4f}, steps: {len(sigmas)-1}")
            use_sigmas = sigmas
        else:
            # Generate LTX-compatible sigmas internally
            print(f"[LTX2Efficient] Generating LTX FlowMatch sigmas (max_shift={max_shift}, base_shift={base_shift}, terminal={terminal})")
            use_sigmas = self._generate_ltx_sigmas(steps, latent_shape, max_shift, base_shift, terminal, stretch)
            print(f"[LTX2Efficient] Generated sigmas: {use_sigmas[0]:.4f} -> {use_sigmas[len(use_sigmas)-2]:.4f} -> {use_sigmas[-1]:.4f}")
        
        # Inject frame_rate into conditioning (same as LTXVConditioning node)
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})
        print(f"[LTX2Efficient] Set frame_rate={frame_rate} in conditioning")
        
        # Store context slice method for patcher
        self._context_slice_method = context_slice_method
        
        # NOTE: For combined audio-video latents from LTXVConcatAVLatent, use LTX2SeparateAVLatent
        # node first to extract video-only latent, then recombine with LTX2CombineAVLatent after.
        
        # Initialize optimization engine
        engine = None
        if optimization_engine != "None":
            engine_config = EngineConfig(
                cache_ratio=engine_cache_ratio,
                merge_ratio=engine_merge_ratio,
                verbose=False,
            )
            engine = get_engine(optimization_engine, engine_config)
            if engine:
                engine.setup(model, steps)
                print(f"[LTX2Efficient] Using optimization engine: {engine.name}")
        
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
        
        # LTX Video latent shape can be:
        # - 5D: (Batch, Channels, Frames, Height, Width) - standard LTX format
        # - 4D: (Frames, Channels, Height, Width) - some sources
        # ComfyUI/LTXVConcatAVLatent may use NestedTensor wrapper
        
        # Convert NestedTensor to regular tensor if needed
        # Try multiple methods since NestedTensor API varies by PyTorch version
        is_nested = str(type(samples).__name__) == 'NestedTensor' or 'nested' in str(type(samples)).lower()
        converted = False  # Track if NestedTensor was successfully converted
        
        if is_nested:
            print(f"[LTX2Efficient] Detected NestedTensor, attempting conversion...")
            
            # Method 1: Try .values() 
            if hasattr(samples, 'values') and callable(samples.values):
                try:
                    samples = samples.values()
                    converted = True
                    print(f"[LTX2Efficient] Converted via .values()")
                except:
                    pass
            
            # Method 2: Try ._values
            if not converted and hasattr(samples, '_values'):
                try:
                    samples = samples._values
                    converted = True
                    print(f"[LTX2Efficient] Converted via ._values")
                except:
                    pass
            
            # Method 3: Try .to_padded_tensor() - common NestedTensor method
            if not converted and hasattr(samples, 'to_padded_tensor'):
                try:
                    samples = samples.to_padded_tensor(0.0)
                    converted = True
                    print(f"[LTX2Efficient] Converted via .to_padded_tensor()")
                except:
                    pass
            
            # Method 4: Try unbind and stack - works for list-like NestedTensors
            if not converted and hasattr(samples, 'unbind'):
                try:
                    tensors = samples.unbind()
                    samples = torch.stack(tensors)
                    converted = True
                    print(f"[LTX2Efficient] Converted via .unbind() + stack")
                except:
                    pass
            
            # Method 5: Direct clone - sometimes works
            if not converted:
                try:
                    samples = samples.clone().detach()
                    converted = True
                    print(f"[LTX2Efficient] Converted via .clone().detach()")
                except:
                    pass
            
            if not converted:
                print(f"[LTX2Efficient] Warning: Could not convert NestedTensor, proceeding with original")
        
        # Ensure we have a contiguous tensor for index operations (only if method exists)
        if hasattr(samples, 'is_contiguous') and not samples.is_contiguous():
            samples = samples.contiguous()
        
        # Get the actual shape
        shape = samples.shape
        ndims = len(shape)
        
        # Check resolution compatibility (LTX requires /32 in pixel space -> /4 in latent space)
        # Latent shape (usually): (..., H, W)
        try:
            h = shape[-2]
            w = shape[-1]
            # VAE usually typically 8x downsample. 
            # LTX patch size is likely 2x2 or 4x4 in latent space.
            # If latent divisible by 1 (all ints), that's trivial.
            # But DiT models work on patches.
            # 768x432 (pixel) -> 96x54 (latent).
            # If patch size 2: 54/2=27. OK.
            # If patch size 4: 54/4=13.5. BAD.
            
            if h % 2 != 0 or w % 2 != 0:
                print(f"[LTX2Efficient] ⚠️ WARNING: Latent dimensions ({w}x{h}) are odd! Video will likely be noisy.")
            elif h % 4 != 0 or w % 4 != 0:
                 print(f"[LTX2Efficient] ⚠️ CRITICAL WARNING: Latent dimensions ({w}x{h}) not divisible by 4!")
                 print(f"[LTX2Efficient] This corresponds to pixel resolution not divisible by 128 (e.g. 448px).")
                 print(f"[LTX2Efficient] LTX generally requires resolutions divisible by 128 (e.g. 512, 640, 768).")
                 print(f"[LTX2Efficient] Your current height {h*32} is likely causing the noise. Try 512 or 384.")
        except:
            pass
        
        # Track if we should skip striding (for unconvertible NestedTensors)
        skip_striding = is_nested and not converted
        
        # If NestedTensor conversion failed, we need to skip frame striding
        # because slicing operations will break the tensor structure
        if skip_striding:
            print(f"[LTX2Efficient] Skipping frame stride for NestedTensor - passing through unchanged")
            keyframes = samples
            original_frames_count = shape[2] if ndims == 5 else (shape[0] if ndims == 4 else 1)
            indices = list(range(original_frames_count))
        else:
            # Normal frame striding for regular tensors
            try:
                if ndims == 5:
                    # Standard LTX format: (B, C, F, H, W)
                    original_frames_count = shape[2]  # Frames at dim 2
                    
                    indices = list(range(0, original_frames_count, frame_stride))
                    keyframes = samples[:, :, indices, :, :]
                    
                elif ndims == 4:
                    # Format: (F, C, H, W) - frames at dim 0
                    original_frames_count = shape[0]
                    
                    indices = list(range(0, original_frames_count, frame_stride))
                    keyframes = samples[indices, :, :, :]
                    
                    # Convert 4D to 5D for LTX model compatibility: (F,C,H,W) -> (1,C,F,H,W)
                    keyframes = keyframes.permute(1, 0, 2, 3).unsqueeze(0)
                    print(f"[LTX2Efficient] Converted 4D tensor to 5D: {keyframes.shape}")
                    
                elif ndims == 3:
                    # Format: (C, H, W) - single frame, expand to 5D
                    print(f"[LTX2Efficient] Single frame 3D tensor detected, expanding to 5D")
                    original_frames_count = 1
                    indices = [0]
                    keyframes = samples.unsqueeze(0).unsqueeze(2)  # (C,H,W) -> (1,C,1,H,W)
                    
                else:
                    raise ValueError(f"[LTX2Efficient] Unexpected latent shape: {shape}. Expected 3D, 4D or 5D tensor.")
                    
            except (IndexError, RuntimeError, TypeError) as e:
                # Final fallback: skip striding entirely
                print(f"[LTX2Efficient] Warning: Indexing failed ({e}). Skipping frame stride...")
                keyframes = samples
                original_frames_count = shape[2] if ndims >= 5 else shape[0] if ndims >= 4 else 1
                indices = list(range(original_frames_count))
        
        print(f"[LTX2Efficient] Keyframe selection: {original_frames_count} frames -> {len(indices)} keyframes (stride={frame_stride})")
        print(f"[LTX2Efficient] Keyframes shape: {keyframes.shape}")
        
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
                
                # Optimization engine step hook
                if engine:
                    engine.step_start(step, x)
                
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
                
                # Engine step end hook
                if engine:
                    engine.step_end(step, x)
            
            # 3. Run Sampler
            # Use KSampler with explicit sigmas for LTX FlowMatch compatibility
            # This matches how SamplerCustomAdvanced operates with LTXVScheduler
            
            # Unwrap latent dictionary to get the tensor
            kf_samples = kf_latent["samples"]
            current_frames = kf_samples.shape[0] if len(kf_samples.shape) == 4 else kf_samples.shape[2]
            print(f"[LTX2Efficient] Sampling on reduced frames: {kf_samples.shape} (Stride={frame_stride})")
            
            # Update patcher with frame info for heuristics
            patcher.set_frame_info(current_frames)
            
            # Prepare noise - comfy.sample.prepare_noise expects a Tensor (latent_image), not a dict
            noise = comfy.sample.prepare_noise(kf_samples, seed)
            
            # Create sampler and get sigmas
            sampler = comfy.samplers.KSampler(
                model, 
                steps=steps,
                device=comfy.model_management.get_torch_device(),
                sampler=sampler_name,
                scheduler="simple",  # This is overridden by explicit sigmas
                denoise=denoise,
                model_options=model.model_options
            )
            
            # Use our FlowMatch sigmas directly
            # Truncate sigmas for denoise < 1.0
            if denoise < 1.0:
                total_steps = len(use_sigmas) - 1
                start_step = int(total_steps * (1 - denoise))
                use_sigmas = use_sigmas[start_step:]
            
            # Run the sampling with explicit sigmas (like SamplerCustomAdvanced)
            print(f"[LTX2Efficient] Running sampling with {len(use_sigmas)-1} steps, sigmas[0]={use_sigmas[0]:.4f}")
            
            # Sample using the sampler's internal sample method with our sigmas
            result_tensor = sampler.sample(
                noise,
                positive, 
                negative, 
                cfg=cfg,
                latent_image=kf_samples,
                start_step=0,
                last_step=len(use_sigmas)-1,
                force_full_denoise=False,
                denoise_mask=None,
                sigmas=use_sigmas,
                callback=step_callback,
                disable_pbar=False,
                seed=seed
            )
            
            # sample() returns a Tensor; wrap it back into a dictionary for the rest of our logic
            result = {"samples": result_tensor}
            
        finally:
            patcher.restore()
            # Cleanup optimization engine
            if engine:
                engine.cleanup()

        # 4. Interpolate
        result_samples = result["samples"]
        
        if interpolation_method == "none":
            print(f"[LTX2Efficient] Interpolation skipped (method='none')")
            full_samples = result_samples
        else:
            print(f"[LTX2Efficient] Interpolating with method: {interpolation_method}")
            
            # Handle 5D (B,C,F,H,W) vs 4D (F,C,H,W) for interpolation
            is_5d = result_samples.ndim == 5
            
            if is_5d:
                # For 5D, we need to interpolate along the frame dimension (dim 2)
                # Squeeze batch, interpolate frames, then unsqueeze
                # Assume batch=1 for now (standard case)
                batch_size = result_samples.shape[0]
                frames_5d = result_samples[0]  # (C, F, H, W)
                frames_transposed = frames_5d.permute(1, 0, 2, 3)  # (F, C, H, W) for interpolation
                full_frames = interpolate(frames_transposed, frame_stride, method=interpolation_method)
                # Transpose back and add batch
                full_frames_5d = full_frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, F, H, W)
                full_samples = full_frames_5d
                
                # Length check for 5D
                full_frames_count = full_samples.shape[2]  # Frames at dim 2
                if full_frames_count > original_frames_count:
                    full_samples = full_samples[:, :, :original_frames_count, :, :]
            else:
                # 4D case (F, C, H, W)
                full_samples = interpolate(result_samples, frame_stride, method=interpolation_method)
                
                # Handle potential length mismatch
                full_frames_count = full_samples.shape[0]
                if full_frames_count > original_frames_count:
                    full_samples = full_samples[:original_frames_count]
        
        # Return video-only latent (use LTX2CombineAVLatent to recombine with audio if needed)
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
            diffusion_model = model_obj # Fallback if no specific attribute
            # print(f"[LTX2Efficient] Warning: Could not find 'diffusion_model' in {type(model_obj)}.")

        self.patched_count = 0
        self.attn1_count = 0
        self.attn2_count = 0
        
        # 1. LTX2-specific patching: look for attn1/attn2 modules
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

        # 2. Patch Top-Level Forward (to fix missing attention_mask)
        # Some LTX models (esp. GGUF loaded) require 'attention_mask' positional arg
        if hasattr(diffusion_model, "forward"):
             self._patch_model_forward(diffusion_model)

    def _patch_model_forward(self, model):
        if model in self.original_forward_maps:
            return
            
        original_forward = model.forward
        self.original_forward_maps[model] = original_forward
        
        def patched_forward(*args, **kwargs):
            # Check if attention_mask is missing from kwargs
            if "attention_mask" not in kwargs:
                kwargs["attention_mask"] = None
            
            # Check for context dimension mismatch (common with Dual Text Encoders/GGUF)
            # LTX expects 4096 dimensions.
            # If input is 7680 (Gemma 3584 + LTX Connector 4096), we need to slice it.
            if "context" in kwargs:
                context = kwargs["context"]
                if hasattr(context, "shape") and len(context.shape) >= 3:
                    dim = context.shape[-1]
                    if dim == 7680:
                        # Get slice method from sampler instance if available
                        slice_method = getattr(self, '_context_slice_method', 'Keep Last 4096 (Slot 2 - LTX Connector)')
                        
                        if "Last 4096" in slice_method or "Slot 2" in slice_method:
                            # Default for Gemma + LTX Connector: use slot 2 (last 4096)
                            kwargs["context"] = context[..., -4096:]
                            if not getattr(self, '_context_logged', False):
                                print(f"[LTX2Patcher] Sliced context 7680->4096 (Last 4096 / LTX Connector Slot 2)")
                                self._context_logged = True
                        elif slice_method == "Auto-detect":
                            # Auto-detect: Check which slot has LTX-compatible embeddings
                            # Gemma (first 3584) typically has different variance pattern
                            first_var = context[..., :4096].var().item()
                            last_var = context[..., -4096:].var().item()
                            gemma_var = context[..., :3584].var().item()
                            
                            # If first 3584 (Gemma portion) has very different variance, Gemma is in slot 1
                            # So LTX Connector is in slot 2
                            if abs(gemma_var - first_var) > 0.05:
                                kwargs["context"] = context[..., -4096:]
                                if not getattr(self, '_context_logged', False):
                                    print(f"[LTX2Patcher] Auto-detected LTX in Slot 2 (last 4096)")
                                    self._context_logged = True
                            elif last_var < first_var:
                                kwargs["context"] = context[..., -4096:]
                                if not getattr(self, '_context_logged', False):
                                    print(f"[LTX2Patcher] Auto-detected LTX in Slot 2 (var={last_var:.4f} < {first_var:.4f})")
                                    self._context_logged = True
                            else:
                                kwargs["context"] = context[..., :4096]
                                if not getattr(self, '_context_logged', False):
                                    print(f"[LTX2Patcher] Auto-detected LTX in Slot 1 (var={first_var:.4f} < {last_var:.4f})")
                                    self._context_logged = True
                        else:
                            # Keep First 4096 (Slot 1 - T5)
                            kwargs["context"] = context[..., :4096]
                            if not getattr(self, '_context_logged', False):
                                print(f"[LTX2Patcher] Sliced context 7680->4096 (First 4096 / T5 Slot 1)")
                                self._context_logged = True
                    elif dim > 4096 and dim != 4096:
                        print(f"[LTX2Patcher] ⚠️ Warning: Context dimension {dim} > 4096. Model might crash if it expects only 4096.")
            
            return original_forward(*args, **kwargs)
            
        model.forward = patched_forward
        print("[LTX2Patcher] Patched top-level model forward to inject 'attention_mask'")

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
