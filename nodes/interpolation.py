"""
Advanced Interpolation Methods for Video Latent Space

Provides multiple interpolation strategies for keyframe interpolation:
1. Linear (lerp) - Simple but can cause flickering
2. Spherical (slerp) - Smooth transitions, good for diffusion latents
3. Motion-Compensated - Estimates motion and warps accordingly
"""

import torch
import torch.nn.functional as F


def interpolate_linear(keyframes: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Simple linear interpolation between keyframes.
    Fast but can cause temporal flickering.
    """
    full = []
    for i in range(len(keyframes) - 1):
        full.append(keyframes[i])
        for j in range(1, stride):
            t = j / stride
            full.append(keyframes[i] * (1 - t) + keyframes[i + 1] * t)
    full.append(keyframes[-1])
    return torch.stack(full)


def interpolate_slerp(keyframes: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Spherical linear interpolation (slerp).
    Better for diffusion latents as it maintains magnitude.
    Treats each frame's latent as a high-dimensional vector.
    """
    full = []
    
    for i in range(len(keyframes) - 1):
        v0 = keyframes[i].flatten()
        v1 = keyframes[i + 1].flatten()
        original_shape = keyframes[i].shape
        
        full.append(keyframes[i])
        
        for j in range(1, stride):
            t = j / stride
            
            # Compute angle between vectors
            v0_norm = v0 / (torch.norm(v0) + 1e-8)
            v1_norm = v1 / (torch.norm(v1) + 1e-8)
            
            dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
            theta = torch.acos(dot)
            
            if theta.abs() < 1e-6:
                # Vectors are nearly parallel, use linear
                interp = v0 * (1 - t) + v1 * t
            else:
                sin_theta = torch.sin(theta)
                interp = (
                    torch.sin((1 - t) * theta) / sin_theta * v0 +
                    torch.sin(t * theta) / sin_theta * v1
                )
            
            full.append(interp.view(original_shape))
    
    full.append(keyframes[-1])
    return torch.stack(full)


def estimate_motion(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """
    Estimate motion between two latent frames using gradient-based method.
    Returns a motion field (dx, dy) for each spatial position.
    
    Works in latent space without requiring decoded images.
    """
    # Average across channels to get spatial structure
    f1 = frame1.mean(dim=0)  # (H, W)
    f2 = frame2.mean(dim=0)  # (H, W)
    
    # Compute spatial gradients of frame1
    # Sobel-like gradients
    grad_x = F.conv2d(
        f1.unsqueeze(0).unsqueeze(0),
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=f1.dtype, device=f1.device).unsqueeze(0).unsqueeze(0) / 8,
        padding=1
    ).squeeze()
    
    grad_y = F.conv2d(
        f1.unsqueeze(0).unsqueeze(0),
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=f1.dtype, device=f1.device).unsqueeze(0).unsqueeze(0) / 8,
        padding=1
    ).squeeze()
    
    # Temporal difference
    dt = f2 - f1
    
    # Optical flow constraint: Ix*dx + Iy*dy + It = 0
    # Simple Lucas-Kanade style estimation
    eps = 1e-6
    grad_sq = grad_x**2 + grad_y**2 + eps
    
    # Estimate flow (simplified - assumes uniform motion locally)
    dx = -dt * grad_x / grad_sq
    dy = -dt * grad_y / grad_sq
    
    # Clamp to reasonable range
    max_flow = 5.0
    dx = torch.clamp(dx, -max_flow, max_flow)
    dy = torch.clamp(dy, -max_flow, max_flow)
    
    return torch.stack([dx, dy], dim=0)  # (2, H, W)


def warp_latent(latent: torch.Tensor, flow: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Warp a latent frame according to flow field.
    
    Args:
        latent: (C, H, W) latent tensor
        flow: (2, H, W) flow field (dx, dy)
        scale: Scale factor for flow (0-1 for interpolation)
    
    Returns:
        Warped latent tensor
    """
    C, H, W = latent.shape
    
    # Create grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=latent.device),
        torch.linspace(-1, 1, W, device=latent.device),
        indexing='ij'
    )
    
    # Add scaled flow to grid (normalize flow to [-1, 1] range)
    flow_x = flow[0] * scale * (2.0 / W)
    flow_y = flow[1] * scale * (2.0 / H)
    
    grid_x = grid_x + flow_x
    grid_y = grid_y + flow_y
    
    # Stack and reshape for grid_sample
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    
    # Warp
    warped = F.grid_sample(
        latent.unsqueeze(0),  # (1, C, H, W)
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    return warped.squeeze(0)  # (C, H, W)


def interpolate_motion_compensated(keyframes: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Motion-compensated interpolation.
    
    Estimates motion between keyframes and uses it to warp intermediate frames,
    producing smoother motion than linear interpolation.
    
    Args:
        keyframes: (N, C, H, W) tensor of keyframe latents
        stride: Number of frames between keyframes
    
    Returns:
        Full frame sequence with interpolated frames
    """
    full = []
    
    for i in range(len(keyframes) - 1):
        frame1 = keyframes[i]
        frame2 = keyframes[i + 1]
        
        full.append(frame1)
        
        # Estimate forward and backward motion
        flow_forward = estimate_motion(frame1, frame2)
        flow_backward = estimate_motion(frame2, frame1)
        
        for j in range(1, stride):
            t = j / stride
            
            # Warp frame1 forward by t
            warped_forward = warp_latent(frame1, flow_forward, scale=t)
            
            # Warp frame2 backward by (1-t)
            warped_backward = warp_latent(frame2, flow_backward, scale=(1 - t))
            
            # Blend warped frames
            blended = warped_forward * (1 - t) + warped_backward * t
            
            full.append(blended)
    
    full.append(keyframes[-1])
    return torch.stack(full)


def interpolate(keyframes: torch.Tensor, stride: int, method: str = "linear") -> torch.Tensor:
    """
    Interpolate between keyframes using specified method.
    
    Args:
        keyframes: (N, C, H, W) tensor of keyframe latents
        stride: Number of frames between keyframes
        method: Interpolation method - "linear", "slerp", or "motion"
    
    Returns:
        Full frame sequence with interpolated frames
    """
    if stride <= 1:
        return keyframes
    
    if len(keyframes) < 2:
        return keyframes
    
    method = method.lower()
    
    if method == "slerp":
        return interpolate_slerp(keyframes, stride)
    elif method == "motion":
        return interpolate_motion_compensated(keyframes, stride)
    else:
        # Default to linear
        return interpolate_linear(keyframes, stride)
