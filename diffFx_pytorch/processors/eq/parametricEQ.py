import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union

from enum import Enum

from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..filters import BiquadFilter


# Parametric Equalizer 
class ParametricEqualizer(ProcessorsBase):
    def __init__(self, sample_rate, num_peak_filters=3):
        self.num_peak_filters = num_peak_filters
        super().__init__(sample_rate)

        # Create peak filters
        self.peak_filters = [
            BiquadFilter(
                sample_rate=self.sample_rate,
                filter_type='pk'
            ) for _ in range(num_peak_filters)
        ]
        
        # Create low shelf and high shelf filters
        self.low_shelf_filter = BiquadFilter(
            sample_rate=self.sample_rate,
            filter_type='ls'
        )
        
        self.high_shelf_filter = BiquadFilter(
            sample_rate=self.sample_rate,
            filter_type='hs'
        )
        
    def _register_default_parameters(self):
        self.params = {
            # Low shelf parameters
            'low_shelf_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
            'low_shelf_frequency': EffectParam(min_val=20.0, max_val=500.0),
            'low_shelf_q_factor': EffectParam(min_val=0.1, max_val=1.0),
            
            # High shelf parameters
            'high_shelf_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
            'high_shelf_frequency': EffectParam(min_val=5000.0, max_val=20000.0),
            'high_shelf_q_factor': EffectParam(min_val=0.1, max_val=1.0),
        }
        
        # Register parameters for each peak filter
        for i in range(self.num_peak_filters):
            peak_name = f'peak_{i+1}'
            self.params.update({
                f'{peak_name}_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
                f'{peak_name}_frequency': EffectParam(min_val=20.0, max_val=20000.0),
                f'{peak_name}_q_factor': EffectParam(min_val=0.1, max_val=10.0),
            })
            
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        
        check_params(norm_params, dsp_params)
        denorm_params = self.map_parameters(norm_params)
        
        # print('norm_params: ', norm_params)
        if norm_params is not None:
            denorm_params = self.map_parameters(norm_params)
            low_shelf_params = {
                'gain_db': denorm_params['low_shelf_gain_db'],
                'frequency': denorm_params['low_shelf_frequency'],
                'q_factor': denorm_params['low_shelf_q_factor']
            }
            
            x_processed = self.low_shelf_filter(x,  dsp_params=low_shelf_params)
            
            # Apply peak filters
            for i in range(self.num_peak_filters):
                peak_name = f'peak_{i+1}'
                peak_params = {
                    'gain_db': denorm_params[f'{peak_name}_gain_db'],
                    'frequency': denorm_params[f'{peak_name}_frequency'],
                    'q_factor': denorm_params[f'{peak_name}_q_factor']
                }
                x_processed = self.peak_filters[i](x_processed, dsp_params=peak_params)
            
            # Apply high shelf filter
            high_shelf_params = {
                'gain_db': denorm_params['high_shelf_gain_db'],
                'frequency': denorm_params['high_shelf_frequency'],
                'q_factor': denorm_params['high_shelf_q_factor']
            }
            x_processed = self.high_shelf_filter(x_processed, dsp_params=high_shelf_params)
        else:
            low_shelf_params = {
                'gain_db': dsp_params['low_shelf_gain_db'],
                'frequency': dsp_params['low_shelf_frequency'],
                'q_factor': dsp_params['low_shelf_q_factor']
            }
            x_processed = self.low_shelf_filter(x, None, low_shelf_params)
            
            # Apply peak filters
            for i in range(self.num_peak_filters):
                peak_name = f'peak_{i+1}'
                peak_params = {
                    'gain_db': params[f'{peak_name}_gain_db'],
                    'frequency': params[f'{peak_name}_frequency'],
                    'q_factor': params[f'{peak_name}_q_factor']
                }
                x_processed = self.peak_filters[i](x_processed, None, peak_params)
            
            # Apply high shelf filter
            high_shelf_params = {
                'gain_db': norm_params['high_shelf_gain_db'],
                'frequency': norm_params['high_shelf_frequency'],
                'q_factor': norm_params['high_shelf_q_factor']
            }
            x_processed = self.high_shelf_filter(x_processed, None, high_shelf_params)
        
        return x_processed
    
    
