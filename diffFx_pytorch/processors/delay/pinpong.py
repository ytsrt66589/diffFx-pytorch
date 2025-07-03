import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.phase import unwrap_phase


class PingPongDelay(ProcessorsBase):
    """Differentiable implementation of a stereo ping-pong delay effect.
    
    This processor implements a stereo delay effect where echoes alternate between
    left and right channels, creating a "ping-pong" spatial pattern. The implementation
    uses a cross-coupled feedback structure in the frequency domain for precise timing
    and smooth transitions.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] Smith, Julius O. "Digital Audio Effects." 
            https://ccrma.stanford.edu/~jos/fp3/Phase_Unwrapping.html
    
    The system is described by coupled transfer functions:

    .. math::

        H_{11}(z) = \\frac{1}{1 - b_1b_2z^{-2N}}

        H_{12}(z) = \\frac{b_1z^{-N}}{1 - b_1b_2z^{-2N}}

        H_{21}(z) = \\frac{b_2z^{-N}}{1 - b_1b_2z^{-2N}}

        H_{22}(z) = \\frac{b_1b_2z^{-2N}}{1 - b_1b_2z^{-2N}}

    where:
        - z^(-N) represents the base delay
        - b1, b2 are feedback gains for each channel
        - System stability ensured by |b1*b2| < 1
    """
    def _register_default_parameters(self):
        """Register delay time, feedback, and mix parameters."""
        self.params = {
            'delay_ms': EffectParam(min_val=0.1, max_val=3000.0),
            'feedback_ch1': EffectParam(min_val=0.0, max_val=0.99),
            'feedback_ch2': EffectParam(min_val=0.0, max_val=0.99),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None , dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the ping-pong delay.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 2, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'delay_ms': Base delay time in milliseconds (0 to 1)
                - 'feedback_ch1': Left channel feedback (0 to 1)
                - 'feedback_ch2': Right channel feedback (0 to 1)
                - 'mix': Wet/dry balance (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify ping-pong parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor of same shape as input. Shape: (batch, 2, samples)
            
        Raises:
            AssertionError: If input is not stereo (2 channels)
        """
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        delay_ms = params['delay_ms'].view(-1, 1, 1)
        b1 = params['feedback_ch1'].view(-1, 1, 1)
        b2 = params['feedback_ch2'].view(-1, 1, 1)
        mix = params['mix'].view(-1, 1, 1)
        
        b, ch, s = x.shape
        assert ch == 2, "Input must be stereo"
        
        max_delay_samples = max(
            1,
            int(torch.max(delay_ms) * self.sample_rate / 1000)
        )
        fft_size = 2 ** int(torch.ceil(torch.log2(torch.tensor(x.shape[-1] + max_delay_samples, device=x.device))))
        pad_right = fft_size - (x.shape[-1] + max_delay_samples)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, pad_right))
        
        X = torch.fft.rfft(x_padded, n=fft_size)
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate, device=x.device)
        phase = -2 * torch.pi * freqs * delay_ms / 1000
        phase = unwrap_phase(phase, dim=-1)
        z_n = torch.exp(1j * phase).to(X.dtype)
        
        eps = 1e-6
        den = 1 - b1 * b2 * z_n * z_n + eps
        
        H11 = 1 / den
        H12 = b1 * z_n / den
        H21 = b2 * z_n / den
        H22 = b1 * b2 * z_n * z_n / den
        
        Y1 = H11 * X[:, 0:1] + H12 * X[:, 1:2]
        Y2 = H21 * X[:, 0:1] + H22 * X[:, 1:2]
        
        Y = torch.cat([Y1, Y2], dim=1)
        y = torch.fft.irfft(Y, n=fft_size)
        y = y[..., max_delay_samples:max_delay_samples + x.shape[-1]]
        
        return (1 - mix) * x + mix * y
    