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

    Args:
    sample_rate (int): Audio sample rate in Hz

    Parameters Details:
    bit_depth: Target bit depth
        - Range: 1.0 to 32.0 bits
        - Lower values create more extreme effects
        - Higher values preserve more detail
        - Integer values match standard bit depths
        - Fractional values allow smooth transitions

    Note:
    - Creates digital degradation effects
    - Useful for lo-fi aesthetics
    - More extreme at lower bit depths
    - Maintains differentiability
    - Automatically handles batch processing

    Examples:
    Basic DSP Usage:
        >>> # Create a bit crusher
        >>> crusher = BitCrusher(sample_rate=44100)
        >>> # Process with moderate bit reduction
        >>> output = crusher(input_audio, dsp_params={
        ...     'bit_depth': 8.0  # 8-bit quality
        ... })

    Neural Network Control:
        >>> # Create a simple bit depth controller
        >>> class BitCrushController(nn.Module):
        ...     def __init__(self, input_size):
        ...         super().__init__()
        ...         self.net = nn.Sequential(
        ...             nn.Linear(input_size, 32),
        ...             nn.ReLU(),
        ...             nn.Linear(32, 1),
        ...             nn.Sigmoid()  # Ensures output is in [0,1] range
        ...         )
        ...     
        ...     def forward(self, x):
        ...         return self.net(x)
        >>> 
        >>> # Process with predicted bit depth
        >>> controller = BitCrushController(input_size=16)
        >>> bit_depth = controller(features)
        >>> output = crusher(input_audio, norm_params={'bit_depth': bit_depth})
    """
    def __init__(self, sample_rate):
        super().__init__(sample_rate)
    
    def _register_default_parameters(self):
        """Register bit depth parameter.
   
        Sets up:
            bit_depth: Target bit depth (1.0 to 32.0)
        """
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
            
        Note:
            Automatically handles broadcasting of bit_depth parameter.
        """

        # Reshape bit_depth for broadcasting
        bit_depth = bit_depth.view(-1, 1, 1)  # [batch, 1, 1]
        
        # Bit depth reduction
        steps = 2 ** bit_depth
        x_crush = torch.round(x * steps) / steps
        
        return x_crush
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the bit crusher.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain:
                    - 'bit_depth': Target bit depth (0 to 1)
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
        
        # Extract parameters
        bit_depth = params['bit_depth']
        
        # Apply bit crushing
        x_crushed = self._bit_crush(x, bit_depth)
        
        return x_crushed