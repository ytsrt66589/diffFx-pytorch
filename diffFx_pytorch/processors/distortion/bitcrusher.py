import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params


class BitCrusher(ProcessorsBase):
    """Differentiable implementation of a bit depth reduction effect. 

    This processor implements bit crushing by reducing the number of available amplitude 
    levels in the signal, creating characteristic digital distortion and quantization
    effects. The implementation uses rounding to integer steps determined by the bit depth.

    The bit reduction process follows:

    .. math::

        y = round(x * 2^{bits}) / 2^{bits}

    where:
        - x is the input signal (assumed to be in [-1, 1] range)
        - bits is the target bit depth
        - round() quantizes to nearest integer step
    """
    def __init__(self, sample_rate, param_range: Dict[str, EffectParam] = None):
        super().__init__(sample_rate, param_range)
    
    def _register_default_parameters(self):
        """Register bit depth parameter."""
        self.params = {
            'bit_depth': EffectParam(min_val=1.0, max_val=32.0)
        }
    
    def _bit_crush(self, x: torch.Tensor, bit_depth: torch.Tensor) -> torch.Tensor:
        """Apply bit depth reduction to the input signal.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            bit_depth (torch.Tensor): Target bit depth. Shape: (batch,)
            
        Returns:
            torch.Tensor: Bit-crushed audio tensor of same shape as input
        """
        bit_depth = bit_depth.view(-1, 1, 1)  
        steps = 2 ** bit_depth
        x_crush = torch.round(x * steps) / steps
        
        return x_crush
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the bit crusher.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain:
                    - bit_depth: Target bit depth (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify bit depth as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Bit-crushed audio tensor of same shape as input
        """
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        bit_depth = params['bit_depth']
        x_crushed = self._bit_crush(x, bit_depth)
        
        return x_crushed