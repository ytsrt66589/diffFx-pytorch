import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.phase import unwrap_phase

# MultiTapDelay
class MultiTapsDelay(ProcessorsBase):
    """Differentiable implementation of a multi-tap delay effect.
    
    The implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    
    This processor implements a parallel delay structure with multiple taps, where each tap
    represents an independent echo with its own delay time and gain. The implementation uses
    frequency-domain processing for precise timing control and efficient computation.

    The transfer function is a sum of delayed signals:

    .. math::

        H(\\omega) = \\sum_{i=0}^{N-1} g_i e^{-j\\omega\\tau_i}

    where:
        - N is the number of taps
        - g_i is the gain of tap i
        - τ_i is the delay time of tap i
        - Phase is unwrapped for each tap

    Args:
        sample_rate (int): Audio sample rate in Hz
        num_taps (int): Number of independent delay taps. Defaults to 4.

    Parameters Details:
        For each tap i (where i ranges from 0 to num_taps-1):
            i_tap_delays_ms: Delay time for tap i
                - Range: 50.0 to 500.0 milliseconds
                - Controls timing of each echo
                - Independent control per tap
                
            i_tap_gains: Gain for tap i
                - Range: 0.0 to 1.0
                - Controls amplitude of each echo
                - Allows creation of complex patterns
                
        mix: Overall wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only original signal
            - 1.0: Only processed signal
    
    Examples:
        Basic DSP Usage:
            >>> # Create a 4-tap delay
            >>> delay = MultiTapsDelay(sample_rate=44100, num_taps=4)
            >>> # Process with rhythmic pattern
            >>> params = {
            ...     '0_tap_delays_ms': 125.0,  # Eighth note at 120 BPM
            ...     '0_tap_gains': 0.8,
            ...     '1_tap_delays_ms': 250.0,  # Quarter note
            ...     '1_tap_gains': 0.6,
            ...     '2_tap_delays_ms': 375.0,  # Dotted quarter
            ...     '2_tap_gains': 0.4,
            ...     '3_tap_delays_ms': 500.0,  # Half note
            ...     '3_tap_gains': 0.2,
            ...     'mix': 0.5
            ... }
            >>> output = delay(input_audio, dsp_params=params)

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class MultiTapController(nn.Module):
            ...     def __init__(self, input_size, num_taps):
            ...         super().__init__()
            ...         num_params = 2 * num_taps + 1  # delays, gains, and mix
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
    def __init__(self, sample_rate, num_taps=4):
        self.num_taps = num_taps
        super().__init__(sample_rate)
        
    def _register_default_parameters(self):
        """Register parameters for all taps and mix.
        
        Creates parameters for each tap:
            - i_tap_delays_ms: Delay time (50.0 to 500.0 ms)
            - i_tap_gains: Tap gain (0.0 to 1.0)
        Plus overall mix parameter.
        
        Total parameters = 2 * num_taps + 1
        """
        self.params = {}
        for i in range(self.num_taps):
            self.params.update({
                f'{i}_tap_delays_ms': EffectParam(min_val=50.0, max_val=500.0),
                f'{i}_tap_gains': EffectParam(min_val=0.0, max_val=1.0)
            })
        self.params['mix'] = EffectParam(min_val=0.0, max_val=1.0)

    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the multi-tap delay.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'base_time': Base delay time for first tap (0 to 1)
                - 'time_mult': Multiplier between successive taps (0 to 1)
                - 'decay': Amplitude decay rate across taps (0 to 1)
                - 'spread': Stereo spread between taps (0 to 1)
                - 'mix': Wet/dry balance (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify multi-tap parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
        """
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
            
        tap_delays_ms = torch.stack([params[f'{i}_tap_delays_ms'] for i in range(self.num_taps)])
        tap_gains = torch.stack([params[f'{i}_tap_gains'] for i in range(self.num_taps)])
        mix = params['mix'].view(-1, 1, 1)
        
        b, ch, s = x.shape
        max_delay_samples = int(torch.max(tap_delays_ms) * self.sample_rate / 1000)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, 0))
        
        X = torch.fft.rfft(x_padded)
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate).to(x.device)
        
        y = torch.zeros_like(X)
        for i in range(self.num_taps):
            phase = -2 * np.pi * freqs * tap_delays_ms[i].view(-1, 1, 1) / 1000
            phase = unwrap_phase(phase, dim=-1).double()
            z_n = torch.exp(1j * phase).to(X.dtype)
            y += tap_gains[i].view(-1, 1, 1) * z_n * X
        
        y = torch.fft.irfft(y, n=x_padded.shape[-1])[:, :, max_delay_samples:]
        return (1 - mix) * x + mix * y

