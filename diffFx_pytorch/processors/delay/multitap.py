import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.phase import unwrap_phase


class MultiTapDelay(ProcessorsBase):
    """Differentiable implementation of a multi-tap delay effect.
    
    This processor implements a parallel delay structure with multiple taps, where each tap
    represents an independent echo with its own delay time and gain. The implementation uses
    frequency-domain processing for precise timing control and efficient computation.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] Smith, Julius O. "Digital Audio Effects." 
            https://ccrma.stanford.edu/~jos/fp3/Phase_Unwrapping.html
    
    The transfer function is a sum of delayed signals:

    .. math::

        H(\\omega) = \\sum_{i=0}^{N-1} g_i e^{-j\\omega\\tau_i}

    where:
        - N is the number of taps
        - g_i is the gain of tap i
        - τ_i is the delay time of tap i
        - Phase is unwrapped for each tap
    """
    def __init__(self, sample_rate, param_range=None, num_taps=4):
        self.num_taps = num_taps
        super().__init__(sample_rate, param_range)
        
    def _register_default_parameters(self):
        """Register parameters for all taps and mix."""
        self.params = {}
        for i in range(self.num_taps):
            self.params.update({
                f'{i}_tap_delays_ms': EffectParam(min_val=50.0, max_val=500.0),
                f'{i}_tap_gains': EffectParam(min_val=0.0, max_val=1.0)
            })
        self.params['mix'] = EffectParam(min_val=0.0, max_val=1.0)

    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the multi-tap delay.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - '{i}_tap_delays_ms': Base delay time for each tap (0 to 1)
                - '{i}_tap_gains': Tap gain for each tap (0 to 1)
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
        
        
        max_delay_samples = max(
            1,
            int(torch.max(tap_delays_ms) * self.sample_rate / 1000)
        )
        fft_size = 2 ** int(torch.ceil(torch.log2(torch.tensor(x.shape[-1] + max_delay_samples))))
        pad_right = fft_size - (x.shape[-1] + max_delay_samples)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, pad_right))
        X = torch.fft.rfft(x_padded, n=fft_size)
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate, device=x.device)
        
        # Vectorized processing for all taps
        delays_expanded = tap_delays_ms[:, :, None, None]  # (num_taps, batch, 1, 1)
        gains_expanded = tap_gains[:, :, None, None]       # (num_taps, batch, 1, 1)
        freqs_expanded = freqs[None, None, None, :]        # (1, 1, 1, freq_bins)
        X_expanded = X[None, :, :, :]                      # (1, batch, channels, freq_bins)

        phases = -2 * torch.pi * freqs_expanded * delays_expanded / 1000  # (num_taps, batch, 1, freq_bins)
        phases = unwrap_phase(phases, dim=-1)
        z_n = torch.exp(1j * phases).to(X.dtype)  # (num_taps, batch, 1, freq_bins)

        # Apply gain, phase, and sum over taps
        y = (gains_expanded * z_n * X_expanded).sum(dim=0)  # (batch, channels, freq_bins)
        
        y = torch.fft.irfft(y, n=fft_size)
        y = y[..., max_delay_samples:max_delay_samples + x.shape[-1]]
        return (1 - mix) * x + mix * y

