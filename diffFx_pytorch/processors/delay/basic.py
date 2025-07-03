import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.phase import unwrap_phase

class BasicDelay(ProcessorsBase):
    """Differentiable implementation of a single-tap delay line.
    
    This processor implements a basic digital delay line using frequency-domain processing
    for precise, artifact-free time delays. It creates a single echo of the input signal
    with controllable delay time and mix level.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] Smith, Julius O. "Digital Audio Effects." 
            https://ccrma.stanford.edu/~jos/fp3/Phase_Unwrapping.html
    
    The delay is implemented in the frequency domain using the time-shift property:

    .. math::
    
        Y(\\omega) = X(\\omega)e^{-j\\omega\\tau}

    where:
        - X(ω) is the input spectrum
        - Y(ω) is the delayed spectrum
        - τ is the delay time in seconds
        - Phase is unwrapped to ensure continuous delay response
    """
    def _register_default_parameters(self):
        """Register delay time and mix parameters."""
        self.params = {
            'delay_ms': EffectParam(min_val=10, max_val=1000.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the delay line.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'delay_ms': Delay time in milliseconds (0 to 1)
                - 'mix': Wet/dry balance (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify delay parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
        """
        # Set proper configuration
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        delay_ms, mix = params['delay_ms'], params['mix']
        
        max_delay_samples = max(
            1,
            int(torch.max(delay_ms) * self.sample_rate / 1000)
        )
        fft_size = 2 ** int(torch.ceil(torch.log2(torch.tensor(x.shape[-1] + max_delay_samples))))
        pad_right = fft_size - (x.shape[-1] + max_delay_samples)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, pad_right))
        X = torch.fft.rfft(x_padded, n=fft_size)
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate, device=x.device)
        phase = -2 * torch.pi * freqs * delay_ms.view(-1, 1, 1) / 1000
        phase = unwrap_phase(phase, dim=-1)
        X_delayed = X * torch.exp(1j * phase).to(X.dtype)
        x_delayed = torch.fft.irfft(X_delayed, n=fft_size)
        x_delayed = x_delayed[..., max_delay_samples:max_delay_samples + x.shape[-1]]

        mix = mix.unsqueeze(-1).unsqueeze(-1)
        return (1 - mix) * x + mix * x_delayed

class BasicFeedbackDelay(ProcessorsBase):
    """Differentiable implementation of a feedback delay line.
    
    This processor implements a delay line with feedback and feedforward paths, creating
    multiple decaying echoes. The implementation uses frequency-domain processing and 
    a feedback-feedforward structure for flexible echo patterns.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] Smith, Julius O. "Digital Audio Effects." 
            https://ccrma.stanford.edu/~jos/fp3/Phase_Unwrapping.html
    
    The transfer function of the system is from [1]:

    .. math::

        H(z) = \\frac{z^{-N} + g_{ff} - g_{fb}}{z^{-N} - g_{fb}}

    where:
        - z^(-N) represents the delay of N samples
        - g_ff is the feedforward gain
        - g_fb is the feedback gain
        - System stability is ensured by limiting |g_fb| < 1
    """
    def _register_default_parameters(self):
        """Register delay, mix, and gain parameters."""
        self.params = {
            'delay_ms': EffectParam(min_val=0.1, max_val=1000.0),
            'mix': EffectParam(min_val=0, max_val=1.0),
            'fb_gain': EffectParam(min_val=0.0, max_val=0.99),
            'ff_gain': EffectParam(min_val=0.0, max_val=0.99)
        }
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None , dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the feedback delay line.

        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'delay_ms': Base delay time in milliseconds (0 to 1)
                - 'fb_gain': Amount of signal fed back through delay line (0 to 1)
                - 'ff_gain': Feedforward gain (0 to 1)
                - 'mix': Wet/dry balance (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify feedback delay parameters as:
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
        
        delay_ms = params['delay_ms'].view(-1, 1, 1)
        g_fb = params['fb_gain'].view(-1, 1, 1)
        g_fb = torch.clamp(g_fb, -0.99, 0.99)
        g_ff = params['ff_gain'].view(-1, 1, 1)
        mix = params['mix'].view(-1, 1, 1)
        
        max_delay_samples = max(
            1,
            int(torch.max(delay_ms) * self.sample_rate / 1000)
        )
        fft_size = 2 ** int(torch.ceil(torch.log2(torch.tensor(x.shape[-1] + max_delay_samples))))
        pad_right = fft_size - (x.shape[-1] + max_delay_samples)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, pad_right))
        X = torch.fft.rfft(x_padded, n=fft_size)
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate, device=x.device)
        phase = -2 * torch.pi * freqs * delay_ms / 1000
        phase = unwrap_phase(phase, dim=-1)
        z_n = torch.exp(1j * phase).to(X.dtype)
        
        eps = 1e-6
        H = (z_n + g_ff - g_fb) / (z_n - g_fb + eps)
        X_delayed = X * H
        
        x_delayed = torch.fft.irfft(X_delayed, n=fft_size)
        x_delayed = x_delayed[..., max_delay_samples:max_delay_samples + x.shape[-1]]
        
        return (1 - mix) * x + mix * x_delayed

class SlapbackDelay(BasicDelay):
    """Differentiable implementation of a slapback delay effect.
    
    The implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] Smith, Julius O. "Digital Audio Effects." 
            https://ccrma.stanford.edu/~jos/fp3/Phase_Unwrapping.html
    
    This processor extends BasicDelay to create a specialized short delay effect
    that emulates the distinctive "doubling" sound popularized in 1950s recordings.
    The delay time range is specifically restricted to create the characteristic
    slapback effect.

    The processor uses the same frequency-domain implementation as BasicDelay:

    .. math::

        Y(\\omega) = X(\\omega)e^{-j\\omega\\tau}

    where τ is restricted to 40-120ms for the slapback effect.
    """
    def _register_default_parameters(self):
        """Register parameters with slapback-specific ranges."""
        self.params = {
            'delay_ms': EffectParam(min_val=40.0, max_val=120.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
    
