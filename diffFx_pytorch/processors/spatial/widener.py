import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union

from enum import Enum

from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..core.midside import * 

class Widener(ProcessorsBase):
    def _register_default_parameters(self):
        # 0.0 -> mono 0.5 -> no change 1.0 -> stereo
        self.params = {
            'width': EffectParam(min_val=0.0, max_val=1.0),
        } 
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        
        check_params(norm_params, dsp_params)
        
        # get parameters 
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        width = params['width']
        bs, chs, seq_len = x.size()
        assert chs == 2, "Input tensor must have shape (bs, 2, seq_len)"
        
        x_ms = lr_to_ms(x, mult=np.sqrt(2)) 
        
        # Split M/S signals
        m, s = torch.split(x_ms, (1, 1), -2)
        
        # Adjust side signal based on width
        # width = 0.0 -> side * 0 = mono
        # width = 0.5 -> side * 1 = original stereo
        # width = 1.0 -> side * 2 = wider stereo
        width = width.view(-1, 1, 1)
        mid = m * (2 * (1 - width))
        side = s * (2 * width)
        
        # Recombine M/S
        x_ms = torch.cat([mid, side], -2)
        x_lr = ms_to_lr(x_ms)
        
        return x_lr