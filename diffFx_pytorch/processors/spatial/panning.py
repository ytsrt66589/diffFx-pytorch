import torch 
import numpy as np 
from typing import Dict, Union
from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..core.midside import * 

class StereoPanning(ProcessorsBase):
    """Differentiable implementation of constant-power stereo panning.
    
    This processor implements stereo panning using a constant-power (equal-power) panning law,
    which maintains consistent perceived loudness across the stereo field. It converts mono
    input signals to stereo by applying complementary gain coefficients to create the desired
    stereo position.

    The panning uses a sinusoidal/cosine-based gain law that ensures:
        - Constant total power across all pan positions
        - Smooth transitions between channels
        - -3dB center attenuation for optimal power distribution

    The gain calculations follow:

    .. math::

        g_L = \\sqrt{\\frac{\\pi/2 - \\theta}{\\pi/2}} \\cos(\\theta)

        g_R = \\sqrt{\\frac{\\theta}{\\pi/2}} \\sin(\\theta)

    where:
        - θ is the panning angle (0 to π/2)
        - g_L is the gain coefficient for left channel
        - g_R is the gain coefficient for right channel
    """
    def _register_default_parameters(self):
        """Register the panning parameter."""
        self.params = {
            'pan': EffectParam(min_val=0.0, max_val=1.0),
        } 
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the stereo panner.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 1, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'pan': Stereo position from left to right (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify panner parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor. Shape: (batch, 2, samples)
            
        Raises:
            AssertionError: If input is not mono (single channel)
        """
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        pan = params['pan']
        bs, chs, seq_len = x.size()
        assert chs == 1, "Input tensor must have shape (bs, 1, seq_len)"
        
        theta = pan * (np.pi / 2)
        left_gain = torch.sqrt(((np.pi / 2) - theta) * (2 / np.pi) * torch.cos(theta))
        right_gain = torch.sqrt(theta * (2 / np.pi) * torch.sin(theta))
        x = x.repeat(1, 2, 1)
        left_gain = left_gain.view(bs, 1, 1)
        right_gain = right_gain.view(bs, 1, 1)
        gains = torch.cat((left_gain, right_gain), dim=1)
        x *= gains

        return x