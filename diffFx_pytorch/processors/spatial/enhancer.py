import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union

from enum import Enum

from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..core.midside import * 
from ..core.phase import unwrap_phase

# Haas Effect 
class StereoEnhancer(ProcessorsBase):
    """Differentiable implementation of stereo enhancement using the Haas effect.
    
    This processor implements stereo enhancement using the Haas effect (precedence effect),
    which creates an enhanced sense of stereo width by introducing small time delays between
    channels. The implementation combines mid-side processing with frequency-domain delay
    to achieve precise control over the stereo image.

    The Haas effect exploits the human auditory system's precedence effect, where delays
    between 1-30ms affect spatial perception without creating distinct echoes. The processor
    applies the delay in the frequency domain for artifact-free time shifting.

    Processing Chain:
        1. Convert L/R to M/S representation
        2. Apply frequency-domain delay to side signal
        3. Apply width scaling to delayed side signal
        4. Convert back to L/R representation

    The frequency domain delay is implemented as:

    .. math::

        S_{delayed}(f) = S(f) * e^{-j2\\pi f \\tau}

    where:
        - S(f) is the side signal in frequency domain
        - f is frequency
        - τ is the delay time in seconds
        - Phase is unwrapped to ensure continuous delay
    """
    def _register_default_parameters(self):
        """Register delay and width parameters.
        
        Sets up:
            - delay_ms: Haas effect delay time (0 to 30 ms)
            - width: Enhancement amount (0.0 to 1.0)
        """
        self.params = {
            'delay_ms': EffectParam(min_val=0.0, max_val=30.0),
            'width': EffectParam(min_val=0.0, max_val=1.0)
        }
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None]=None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the stereo enhancer.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 2, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'delay_ms': Delay time for side signal (0 to 1)
                - 'width': Stereo width enhancement (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify enhancer parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor. Shape: (batch, 2, samples)
            
        Raises:
            AssertionError: If input is not stereo (two channels)
        """
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        bs, chs, seq_len = x.size()
        assert chs == 2, "Input tensor must have shape (bs, 2, seq_len)"
        
        # Convert to M/S
        x_ms = lr_to_ms(x, mult=1/np.sqrt(2))
        mid, side = torch.split(x_ms, (1, 1), -2)
        
        # Apply delay to side channel
        # Calculate FFT size (next power of 2 for efficiency)
        max_delay_samples = max(
            1,
            int(torch.max(params['delay_ms']) * self.sample_rate / 1000)
        )
        fft_size = 2 ** int(np.ceil(np.log2(side.shape[-1] + max_delay_samples)))
        # Pad input signal to FFT size
        pad_right = fft_size - (x.shape[-1] + max_delay_samples)
        side_padded = torch.nn.functional.pad(side, (max_delay_samples, pad_right))

        Side = torch.fft.rfft(side_padded, n=fft_size)
        freqs = torch.fft.rfftfreq(side_padded.shape[-1], 1/self.sample_rate).to(x.device)
        
        phase = -2 * np.pi * freqs * params['delay_ms'].view(-1, 1, 1) / 1000
        phase = unwrap_phase(phase, dim=-1)
        Side = Side * torch.exp(1j * phase).to(Side.dtype)
        
        # Convert back to time domain with width control
        side_delayed = torch.fft.irfft(Side, n=fft_size)
        side_delayed = side_delayed[..., max_delay_samples:max_delay_samples + side.shape[-1]]
        width = params['width'].view(-1, 1, 1)
        
        x_ms_new = torch.cat([mid, side_delayed * width], -2)
        x_lr = ms_to_lr(x_ms_new, mult=1/np.sqrt(2))
        
        return x_lr