import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.phase import unwrap_phase

# 4 feedback comb filters + 2 allpass filters 
class SchroederReverb(ProcessorsBase):

    def _register_default_parameters(self):
        self.params = {
            'delay_ms': EffectParam(min_val=10, max_val=3000.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        # Set proper configuration
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        # get parameters 
        delay_ms, mix = params['delay_ms'], params['mix']
        
        # Padding 
        max_delay_samples = max(
            1,
            int(torch.max(delay_ms) * self.sample_rate / 1000)
        )
        # Calculate FFT size (next power of 2 for efficiency)
        fft_size = 2 ** int(np.ceil(np.log2(x.shape[-1] + max_delay_samples)))
        # Pad input signal to FFT size
        pad_right = fft_size - (x.shape[-1] + max_delay_samples)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, pad_right))

        # x_padded = torch.nn.functional.pad(x, (max_delay_samples, 0))
        
        # Convert to frequency domain
        X = torch.fft.rfft(x_padded, n=fft_size)
        
        # Phase calculation with unwrapping
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate).to(x.device)
        phase = -2 * np.pi * freqs * delay_ms.view(-1, 1, 1) / 1000
        phase = unwrap_phase(phase, dim=-1)
        
        # Apply phase shift
        X_delayed = X * torch.exp(1j * phase).to(X.dtype)
        
        # IFFT and trim padding
        # x_delayed = torch.fft.irfft(X_delayed, n=x_padded.shape[-1])#[:, :, max_delay_samples:]
        # Trim to match original input length
        x_delayed = torch.fft.irfft(X_delayed, n=fft_size)
        x_delayed = x_delayed[..., max_delay_samples:max_delay_samples + x.shape[-1]]

        mix = mix.unsqueeze(-1).unsqueeze(-1)
        return (1 - mix) * x + mix * x_delayed