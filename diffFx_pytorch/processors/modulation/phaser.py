import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from torchlpc import sample_wise_lpc

from typing import Optional

import torch
from torch import Tensor as T

# Phaser 
# > Depth 
# > Width 
# > Feedback 
# > LFO Frequency 
def time_varying_fir(x: T, b: T, zi: Optional[T] = None) -> T:
    assert x.ndim == 2
    assert b.ndim == 3
    assert x.size(0) == b.size(0)
    assert x.size(1) == b.size(1)
    order = b.size(2) - 1
    x_padded = F.pad(x, (order, 0))
    if zi is not None:
        assert zi.shape == (x.size(0), order)
        x_padded[:, :order] = zi
    x_unfolded = x_padded.unfold(dimension=1, size=order + 1, step=1)
    x_unfolded = x_unfolded.unsqueeze(3)
    b = b.flip(2).unsqueeze(2)
    y = b @ x_unfolded
    y = y.squeeze(3)
    y = y.squeeze(2)
    return y

def fourth_order_ap_coeffs(p):
    b = torch.stack([p**4, -4 * p**3, 6 * p**2, -4 * p, torch.ones_like(p)], dim=p.ndim)
    a = b.flip(-1)
    return a, b

class Phaser(ProcessorsBase):
    """Differentiable implementation of a high-order allpass phaser effect.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] Yu, Chin-Yun, et al. "Differentiable all-pole filters for time-varying audio systems." 
            arXiv preprint arXiv:2404.07970 (2024).
    
    This processor implements a sophisticated phaser effect using allpass filters 
    with time-varying coefficients, creating complex phase-shifting modulations 
    in the audio signal. The implementation provides precise control over the 
    phasing effect through multiple parameters.

    Processing Chain:
    1. Generate low-frequency oscillator (LFO) modulation
    2. Create time-varying allpass filter coefficients
    3. Apply fourth-order allpass filtering
    4. Mix processed and original signals

    The phaser effect works by creating phase shifts across different frequency 
    bands, resulting in a sweeping, swirling sound characteristic of classic 
    analog phasers. Unlike simple single-stage phasers, this implementation 
    uses a fourth-order allpass filter for richer, more complex modulations.
    
    Mathematical Concept:
    The core of the phaser is a time-varying allpass filter where the transfer 
    function is dynamically modified by a low-frequency oscillator (LFO). The 
    phase shift is controlled by the allpass coefficient p(t), which is derived 
    from the LFO and frequency range parameters.

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.

    Attributes:
        sample_rate (int): Audio sample rate in Hz
        prev_states (torch.Tensor, optional): Previous filter states for stateful processing
        osc (callable): Oscillator function (defaults to torch.sin)
        lpc_func (callable): Linear prediction coding function for filtering

    Parameters Details:
    f0: LFO Frequency
        - Range: 0.1 to 5.0 Hz
        - Controls speed of phase modulation
        - Determines sweeping rate of the effect
        
    f_min: Minimum Cutoff Frequency
        - Range: 100.0 to 2000.0 Hz
        - Lower bound of frequency modulation
        - Defines the start of the phase-shifting range
        
    f_max: Maximum Cutoff Frequency
        - Range: 2000.0 to 10000.0 Hz
        - Upper bound of frequency modulation
        - Defines the end of the phase-shifting range
        
    feedback: Feedback Amount
        - Range: 0.0 to 0.9
        - Controls resonance and intensity of the phasing effect
        - Higher values create more pronounced peaks and notches
        
    wet_mix: Wet/Dry Mix
        - Range: 0.0 to 1.0
        - 0.0: Only clean (dry) signal
        - 1.0: Only processed (wet) signal

    Note:
    The processor supports the following features:
        - Fourth-order allpass filtering
        - Dynamic frequency range modulation
        - Precise feedback control
        - Efficient batch processing
        - Neural network compatible

    Warning:
    When using with neural networks:
        - Ensure norm_params are in range [0, 1]
        - Parameters will be automatically mapped to ranges
        - Network output should be properly normalized
        - Input must be mono or stereo
        - Parameter order must match _register_default_parameters()

    Examples:
    Basic DSP Usage:
        >>> # Create a phaser
        >>> phaser = Phaser(sample_rate=44100)
        >>> # Process with musical settings
        >>> output = phaser(input_audio, dsp_params={
        ...     'f0': 0.5,          # 0.5 Hz LFO
        ...     'f_min': 500.0,     # Start at 500 Hz
        ...     'f_max': 5000.0,    # End at 5000 Hz
        ...     'feedback': 0.6,    # Moderate feedback
        ...     'wet_mix': 0.7      # 70% processed signal
        ... })

    Neural Network Control:
        >>> # Simple parameter prediction network
        >>> class PhaserController(nn.Module):
        ...     def __init__(self, input_size, num_params):
        ...         super().__init__()
        ...         self.net = nn.Sequential(
        ...             nn.Linear(input_size, 32),
        ...             nn.ReLU(),
        ...             nn.Linear(32, num_params),
        ...             nn.Sigmoid()  # Ensures output is in [0,1] range
        ...         )
        ...     
        ...     def forward(self, x):
        ...         return self.net(x)
    """
    def __init__(self, sample_rate):
        super().__init__(sample_rate)
        self.prev_states = None
        self.osc = torch.sin 
        self.lpc_func = sample_wise_lpc
        
    def _register_default_parameters(self):
        """Register default parameters for the phaser effect.
   
        Sets up:
            f0: LFO frequency (0.1 to 5.0 Hz)
            f_min: Minimum cutoff frequency (100.0 to 2000.0 Hz)
            f_max: Maximum cutoff frequency (2000.0 to 10000.0 Hz)
            feedback: Feedback amount (0.0 to 0.9)
            wet_mix: Wet/dry balance (0.0 to 1.0)
        """
        self.params = {
            'f0': EffectParam(min_val=0.1, max_val=5.0),      # LFO frequency (Hz)
            'f_min': EffectParam(min_val=100.0, max_val=2000.0),  # Min cutoff freq (Hz)
            'f_max': EffectParam(min_val=2000.0, max_val=10000.0), # Max cutoff freq (Hz)
            'feedback': EffectParam(min_val=0.0, max_val=0.9),   # Feedback amount
            'wet_mix': EffectParam(min_val=0.0, max_val=1.0),    # Wet/dry mix
        }
        
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the phaser effect.
   
        Args:
            x (torch.Tensor): Input audio tensor. 
                Shape: (batch, channels, samples)
                Supports mono or stereo inputs
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain keys:
                    - 'f0': LFO frequency
                    - 'f_min': Minimum cutoff frequency
                    - 'f_max': Maximum cutoff frequency
                    - 'feedback': Feedback amount
                    - 'wet_mix': Wet/dry mix
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): 
                Direct DSP parameters. Can specify as:
                    - float/int: Single value applied to entire batch
                    - 0D tensor: Single value applied to entire batch
                    - 1D tensor: Batch of values matching input batch size
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor 
            Shape: (batch, channels, samples)
            
        Raises:
            AssertionError: If input tensor dimensions are incorrect
        """
        batch_size, chs, num_samples = x.shape
        device = x.device
        
        # Get parameters exactly as in original code
        # Get parameters
        check_params(norm_params, dsp_params)
        # Set proper configuration
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        f0 = params['f0']
        f_min = params['f_min']
        f_max = params['f_max']
        feedback = params['feedback']
        mix = params['wet_mix']
        
        # Calculate normalized frequencies
        d_min = 2.0 * f_min / self.sample_rate
        d_max = 2.0 * f_max / self.sample_rate
        depth = (d_max - d_min) * 0.5
        
        # Generate time vector
        t = torch.arange(num_samples, device=device) / self.sample_rate
        
        # input_f0 = 2 * torch.pi * f0.view(-1, 1) * t
        # input_f0 = input_f0.unsqueeze(-1)
        input_f0 = 2 * torch.pi * f0.view(-1, 1) * t
        # Generate LFO (using sawtooth wave as in original)
        lfo = d_min.view(-1, 1) + depth.view(-1, 1) * (1.0 + self.osc(input_f0))
        
        # Calculate allpass coefficient
        p = (1.0 - torch.tan(lfo)) / (1.0 + torch.tan(lfo))
        
        # Process each channel
        x = x.squeeze(1)
        
        combine_a, combine_b = fourth_order_ap_coeffs(p)
        combine_denom = combine_a - feedback.view(-1, 1, 1).abs() * combine_b
        combine_b = combine_b / combine_denom[..., :1]
        combine_denom = combine_denom / combine_denom[..., :1]
        
        y_ch = time_varying_fir(x, combine_b)
        y_ch = self.lpc_func(y_ch, combine_denom[..., 1:], None)  
        
        # Mix wet and dry signals
        y = mix.view(-1, 1) * y_ch + (1.0 - mix.view(-1, 1)) * x
            
        return y.unsqueeze(1)