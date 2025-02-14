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
class Enhancer(ProcessorsBase):
    def _register_default_parameters(self):
        self.params = {
            'delay_ms': EffectParam(min_val=0.0, max_val=30.0),
            'width': EffectParam(min_val=0.0, max_val=1.0)
        }
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        bs, chs, seq_len = x.size()
        assert chs == 2, "Input tensor must have shape (bs, 2, seq_len)"
        
        # Convert to M/S
        x_ms = lr_to_ms(x, mult=np.sqrt(2))
        mid, side = torch.split(x_ms, (1, 1), -2)
        
        # Apply delay to side channel
        Side = torch.fft.rfft(side)
        freqs = torch.fft.rfftfreq(x.shape[-1], 1/self.sample_rate).to(x.device)
        
        phase = -2 * np.pi * freqs * params['delay_ms'].view(-1, 1, 1) / 1000
        phase = unwrap_phase(phase, dim=-1)
        Side = Side * torch.exp(1j * phase).to(Side.dtype)
        
        # Convert back to time domain with width control
        side = torch.fft.irfft(Side, n=x.shape[-1])
        width = params['width'].view(-1, 1, 1)
        
        x_ms_new = torch.cat([mid, side * width], -2)
        x_lr = ms_to_lr(x_ms_new)
        
        return x_lr