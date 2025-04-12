import math
import torch
import torch.fft

from typing import Dict, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from .biquad import BiquadFilter
# https://ccrma.stanford.edu/~jos/filters/DC_Blocker.html 
# first-order high-pass filter
class DCFilter(ProcessorsBase):
    def __init__(self, sample_rate=44100, learnable=False):
        super().__init__(sample_rate)
        # Create filter banks
        # self.highpass_filters = IIRFilter(order=2)
        
        self.highpass_filters = BiquadFilter(sample_rate, filter_type='hp')
        
        self.default_dsp_params = {
            'frequency': torch.full((1,), 20.0),
            'q_factor': torch.full((1,), 0.707),
            'gain_db': torch.zeros((1,))
        }
        self.learnable = learnable
        
    def _register_default_parameters(self):
        self.params = {
            'frequency': EffectParam(min_val=0.0, max_val=20.0),  # Crossover frequency
        }
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None]=None, dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.learnable:
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
            

        batch_size = x.shape[0]
        filter_params = {
            'frequency': self.default_dsp_params['frequency'].expand(batch_size),
            'q_factor': self.default_dsp_params['q_factor'].expand(batch_size),
            'gain_db': self.default_dsp_params['gain_db'].expand(batch_size)
        }
        x_processed = self.highpass_filters(x, None, filter_params)
        
        return x_processed