import numpy as np
import torch
import torch.fft
import torch.nn.functional as F


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



def ms_to_z_alpha(time_ms: torch.Tensor, sample_rate) -> torch.Tensor:
    """Convert time constant in milliseconds to z-alpha (pre-sigmoid coefficient).
        
        This function converts a time constant (in ms) to a z-domain coefficient that,
        when passed through a sigmoid function, will produce the correct smoothing
        coefficient for the filter.
        
        The conversion is based on the relationship:
            alpha = sigmoid(z_alpha) = exp(-1/(time_ms * sample_rate / 1000))
        
        Args:
            time_ms (torch.Tensor): Time constant in milliseconds. Should be positive.
                Typically represents attack or release time.
                Shape: (B, 1) or (B,) where B is batch size.
                
        Returns:
            torch.Tensor: Coefficient in z-domain that will produce the desired 
                smoothing coefficient after sigmoid transformation.
                Shape matches input shape.
                
        Notes:
            - For numerical stability, time values are clamped to a minimum of 1e-6 ms
            - The resulting alpha (after sigmoid) will be in range (1e-6, 1-1e-6)
            - Very small time values will result in fast response (alpha near 0)
            - Very large time values will result in slow response (alpha near 1)
    """
    # Ensure positive time values and good numerical behavior
    time_ms = torch.clamp(time_ms, min=1e-6)
    
    # Convert to samples
    samples = time_ms * sample_rate / 1000
    
    # Compute alpha (filter coefficient)
    # Using exp(-1/samples) to get the desired smoothing behavior
    desired_alpha = torch.exp(-1 / samples)
    
    # Clamp for numerical stability before computing z_alpha
    desired_alpha = torch.clamp(desired_alpha, min=1e-6, max=1-1e-6)
    
    # Convert to z-domain using the inverse of sigmoid (logit function)
    # logit(x) = log(x/(1-x)) = log(x) - log(1-x)
    z_alpha = torch.log(desired_alpha) - torch.log(1 - desired_alpha)
    
    return z_alpha