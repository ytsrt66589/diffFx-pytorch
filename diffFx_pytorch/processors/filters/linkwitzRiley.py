import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from ..core.iir import IIRFilter
from ..core.midside import lr_to_ms, ms_to_lr
from ..core.utils import normalize_impulse, freq_to_normalized
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from .biquad import BiquadFilter

class LinkwitzRileyFilter(ProcessorsBase):
    """Linkwitz-Riley crossover filter implementation using cascaded Butterworth filters"""
    def __init__(self, sample_rate=44100, order=4):
        super().__init__(sample_rate)
        self.num_cascades = order // 2  # Support different orders
        # Create filter banks
        self.lowpass_filters = nn.ModuleList([
            BiquadFilter(sample_rate, filter_type='LP') 
            for _ in range(self.num_cascades)
        ])
        self.highpass_filters = nn.ModuleList([
            BiquadFilter(sample_rate, filter_type='HP')
            for _ in range(self.num_cascades)
        ])
        
    def _register_default_parameters(self):
        self.params = {
            'frequency': EffectParam(min_val=20.0, max_val=20000.0),  # Crossover frequency
        }
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        check_params(norm_params, dsp_params)
        
        if dsp_params is not None:
            frequency = dsp_params['frequency']
        else:
            frequency = self.map_parameters(norm_params)['frequency']
        
        # Set Q factor for Butterworth response (Q = 0.707)
        filter_params = {
            'frequency': frequency,
            'q_factor': torch.ones_like(frequency) * 0.707,
            'gain_db': torch.zeros_like(frequency)
        }
        
        # Process through filter banks
        low = x
        high = x
        for lp, hp in zip(self.lowpass_filters, self.highpass_filters):
            low = lp.process(low, norm_params=None, dsp_params=filter_params)
            high = hp.process(high, norm_params=None, dsp_params=filter_params)
        
        return torch.cat((low, high), dim=1)
        