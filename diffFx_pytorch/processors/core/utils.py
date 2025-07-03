import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from torchlpc import sample_wise_lpc

def rms_difference(X, Y, eps=1e-7):
    X_rms = torch.log(X.square().mean((-1, -2)) + eps)
    Y_rms = torch.log(Y.square().mean((-1, -2)) + eps)
    diff = (X_rms - Y_rms).abs().sum()
    return diff


def normalize_impulse(ir, eps=1e-12):
    assert ir.ndim == 3
    e = ir.square().sum(2, keepdim=True).mean(1, keepdim=True)
    ir = ir / torch.sqrt(e + eps)
    return ir


def freq_to_normalized(freq_hz, sample_rate):
    """Convert frequency from Hz to normalized frequency."""
    return 2 * np.pi * freq_hz / sample_rate    



def ms_to_alpha(time_ms: torch.Tensor, sample_rate) -> torch.Tensor:
    """Convert time constant in milliseconds to alpha (coefficient of IIR Filter).
        
        This function converts a time constant (in ms) to a coefficient that,
        when passed through a sigmoid function, will produce the correct smoothing
        coefficient for the filter.
        
        The conversion uses the industry-standard formula:
            alpha = 1 - exp(-2.2 / (time_ms * sample_rate / 1000))
        
        This matches the behavior of traditional analog envelope followers and
        provides the expected attack/release characteristics.
        
        Args:
            time_ms (torch.Tensor): Time constant in milliseconds. Should be positive.
                Typically represents attack or release time.
                Shape: (B, 1) or (B,) where B is batch size.
                
        Returns:
            torch.Tensor: Coefficient of IIR Filter that will produce the desired 
                smoothing coefficient after sigmoid transformation.
                Shape matches input shape.
    """
    desired_alpha = 1 - torch.exp(-2200 / time_ms / sample_rate)
    return desired_alpha



def variable_delay(phase: torch.Tensor, audio: torch.Tensor, buf_size: int) -> torch.Tensor:
    """Variable delay implementation using grid_sample

    Args:
        phase (torch.Tensor): normalized delay time (0~1) (batch, channel, n_samples)
        audio (torch.Tensor): input audio (batch, channel, n_samples)
        buf_size (int): maximum delay buffer size in samples

    Returns:
        torch.Tensor: delayed audio (batch, channel, n_samples)
    """
    batch_size, n_ch, n_samples = audio.shape
    
    # Reshape audio for grid_sample: (batch*channel, 1, 1, n_samples)
    audio_4d = audio.reshape(batch_size * n_ch, 1, 1, n_samples)
    
    # Calculate delay grid
    delay_ratio = buf_size * 2 / n_samples
    grid_x = torch.linspace(-1, 1, n_samples, device=audio.device)
    
    # Expand grid for batch and channel dimensions
    grid_x = grid_x.expand(batch_size, n_ch, n_samples)
    grid_x = grid_x - delay_ratio + delay_ratio * phase  # Apply phase modulation
    
    # Reshape for grid_sample: (batch*channel, 1, n_samples, 2)
    grid_x = grid_x.reshape(batch_size * n_ch, 1, n_samples, 1)
    grid_y = torch.zeros_like(grid_x)
    grid = torch.cat([grid_x, grid_y], dim=-1)
    
    # Apply delay using grid_sample
    output = torch.nn.functional.grid_sample(
        audio_4d, 
        grid,
        align_corners=True,
        mode='bilinear',
        padding_mode='zeros'
    )
    
    # Reshape back to original dimensions
    return output.squeeze(2).squeeze(1).reshape(batch_size, n_ch, n_samples)


