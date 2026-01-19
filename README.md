# LTX2 Efficient Video Sampler

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A high-performance custom node for LTX2 video generation in ComfyUI, designed to **drastically reduce GPU usage, VRAM requirements, and temperature** on consumer GPUs (especially 6GB cards like RTX 2060).

![GPU Temperature](https://img.shields.io/badge/Target-65°C-brightgreen)

## ✨ Features

### 🎯 Core Optimizations

- **Frame-Stride Diffusion**: Process only keyframes (e.g., 1 out of 4), interpolate the rest
- **Spatial Attention Freezing**: Skip redundant attention computation after configurable threshold
- **LTX2-Specific Hooks**: Precise targeting of `attn1` (self-attention) vs `attn2` (cross-attention)

### 🌡️ Thermal Management

- **Real-Time GPU Monitoring**: Uses `pynvml` to read actual GPU temperature
- **Dynamic Throttling**: Auto-adjusts processing speed to maintain target temperature
- **Emergency Protection**: Automatic 2-second pause at critical temps (83°C+)
- **Preset Profiles**: Quick optimization for different GPU tiers

### 🎬 Advanced Interpolation

- **Linear**: Fast, basic blending
- **Slerp**: Spherical interpolation - smoother for diffusion latents
- **Motion-Compensated**: Optical flow-based warping for best motion quality

### 📦 Additional Nodes

- **LTX2 Temporal VAE Decode**: Memory-efficient VAE decoding with temporal tiling
- **LTX2 Separate Audio/Video Latent**: Extract video and audio from combined latents (for use with LTXVConcatAVLatent)
- **LTX2 Combine Audio/Video Latent**: Recombine processed video with audio for downstream nodes
- **LTX2 Empty Audio Latent**: Generate placeholder audio latent matching video duration

### 🎵 Audio-Video Workflow

For workflows using `LTXVConcatAVLatent` or MMAudio:

```
[LTXVConcatAVLatent] → [LTX2 Separate AV] → [LTX2 Efficient Sampler] → [LTX2 Combine AV] → [Decode]
                                         ↓                              ↑
                                  [Audio Pipeline (optional)] ──────────┘
```

## 📋 Requirements

- ComfyUI (latest)
- PyTorch with CUDA
- `pynvml` (optional, for thermal monitoring)

## 🚀 Installation

### Option 1: ComfyUI Manager (Recommended)

Search for "LTX2 Efficient" in ComfyUI Manager and install.

### Option 2: Manual Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/kakachiex2/LTX2-Efficient-Sampler
cd comfyui-ltx2-efficient
pip install -r requirements.txt
```

## 🎛️ Node Settings

### Optimization Presets

| Preset                      | Freeze Ratio | Throttle | Frame Stride | Best For        |
| --------------------------- | ------------ | -------- | ------------ | --------------- |
| Performance (RTX 3080+)     | 70%          | 0ms      | 2            | High-end GPUs   |
| Balanced (RTX 3060/3070)    | 50%          | 25ms     | 4            | Mid-range       |
| Aggressive (RTX 2060/2070)  | 30%          | 50ms     | 4            | 6GB VRAM        |
| Ultra Low Power (GTX 1660)  | 15%          | 100ms    | 6            | Older GPUs      |
| Extreme Cool (RTX 2060 6GB) | 10%          | 150ms    | 8            | Target 65°C     |
| Thermal Auto-Scale          | 30%          | Dynamic  | 4            | **Recommended** |
| Custom                      | Manual       | Manual   | Manual       | Fine-tuning     |

### Key Parameters

| Parameter              | Default            | Description                                 |
| ---------------------- | ------------------ | ------------------------------------------- |
| `optimization_preset`  | Thermal Auto-Scale | Quick optimization profile                  |
| `target_temp`          | 70°C               | Target GPU temperature (for auto-scale)     |
| `frame_stride`         | 4                  | Process every Nth frame (higher = faster)   |
| `freeze_ratio`         | 0.3                | Start freezing attention at this % of steps |
| `attention_window`     | 4                  | Temporal attention window size              |
| `interpolation_method` | slerp              | linear, slerp, or motion                    |

### Interpolation Methods

| Method   | Speed     | Quality | Best For                         |
| -------- | --------- | ------- | -------------------------------- |
| `linear` | ⚡ Fast   | Basic   | Quick previews                   |
| `slerp`  | ⚡ Fast   | Smooth  | **Default** - diffusion latents  |
| `motion` | 🐢 Slower | Best    | Videos with significant movement |

## 📖 Usage

1. Add **LTX2 Efficient Video Sampler** node
2. Connect your model, latent video, and conditioning
3. Select an **Optimization Preset** (start with "Thermal Auto-Scale")
4. Set **Target Temp** to your comfort level (65-70°C recommended)
5. Choose **Interpolation Method** (slerp is a good default)
6. Run and watch console for temperature feedback!

### Console Output

```
[LTX2Efficient] Using preset: Thermal Auto-Scale
[LTX2Efficient] -> THERMAL AUTO-SCALE enabled, target_temp=70°C
[LTX2Efficient] GPU Monitor active. Current temp: 47°C
[LTX2Efficient] Patched 96 attention layers (attn1=48, attn2=48)
[LTX2Efficient] Step 5: Temp=52°C, Delay=0ms
[LTX2Efficient] Freezing spatial at step 6 (30.0%)
[LTX2Efficient] Interpolating with method: slerp
```

## 🔧 Troubleshooting

### "pynvml not installed" warning

```bash
pip install pynvml
```

Or install `nvidia-ml-py` (pynvml is deprecated but still works).

### Still running hot?

1. Lower `target_temp` to 60-65
2. Use "Extreme Cool" preset
3. Increase `frame_stride` to 6-8
4. Lower video resolution/frame count

### Quality issues?

1. Try `interpolation_method: motion`
2. Lower `frame_stride` to 2-3
3. Increase `freeze_ratio` to 0.5+

## 🏗️ Architecture

```
comfyui-ltx2-efficient/
├── __init__.py              # Node registration
├── nodes/
│   ├── ltx2_efficient_sampler.py  # Main sampler node
│   ├── ltx2_vae_decode.py         # Temporal VAE decode
│   ├── gpu_monitor.py             # pynvml temperature monitoring
│   └── interpolation.py           # Linear/Slerp/Motion interpolation
├── frontend/                 # Vue UI components (WIP)
└── Reference/               # Technical documentation
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The amazing UI framework
- [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video) - The video model
- Inspired by research on efficient video diffusion

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Made with ❤️ for the ComfyUI community**
