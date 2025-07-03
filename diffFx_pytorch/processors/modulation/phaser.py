import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union, Optional
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..filters import BiquadFilter

import torch
from torch import Tensor as T


class Phaser(ProcessorsBase):
    """Simple differentiable phaser using cascaded all-pass biquad filters."""
    def __init__(self, sample_rate=44100, param_range=None, num_stages=4):
        super().__init__(sample_rate, param_range)
        self.num_stages = num_stages
        self.stages = nn.ModuleList([
            BiquadFilter(sample_rate, filter_type='ap')
            for _ in range(num_stages)
        ])

    def _register_default_parameters(self):
        self.params = {
            'frequency': EffectParam(min_val=300.0, max_val=2000.0),  # All-pass center freq
            'q_factor': EffectParam(min_val=0.1, max_val=1.0),        # All-pass Q
            'feedback': EffectParam(min_val=0.0, max_val=0.9),        # Feedback
            'mix': EffectParam(min_val=0.0, max_val=1.0),             # Wet/dry mix
        }

    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params

        batch, channels, samples = x.shape
        wet = x
        feedback = params['feedback'].view(-1, 1, 1)
        for stage in self.stages:
            stage_params = {
                'frequency': params['frequency'],
                'q_factor': params['q_factor'],
                'gain_db': torch.zeros_like(params['frequency'])  
            }
            wet = stage(wet, None, dsp_params=stage_params)
            wet = wet + feedback * x

        mix = params['mix'].view(-1, 1, 1)
        y = mix * wet + (1.0 - mix) * x
        return y


