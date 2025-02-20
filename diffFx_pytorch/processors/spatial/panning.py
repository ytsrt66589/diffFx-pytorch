import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union

from enum import Enum

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

    Args:
        sample_rate (int): Audio sample rate in Hz

    Parameters Details:
        pan: Panning position control
            - 0.0: Full left
            - 0.5: Center
            - 1.0: Full right
            - Controls the perceived position in the stereo field
            - Mapped internally to panning angle θ

    Note:
        - Input must be mono (single channel)
        - Output is always stereo (two channels)
        - Total power is preserved across all pan positions
        - Uses equal-power (constant-power) panning law

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameter will be automatically mapped to pan position
            - Ensure your network output is properly normalized (e.g., using sigmoid)

    Examples:
        Basic DSP Usage:
            >>> # Create a stereo panner
            >>> panner = StereoPanning(sample_rate=44100)
            >>> # Process mono audio with direct panning
            >>> output = panner(input_audio, dsp_params={
            ...     'pan': 0.75  # Pan 75% to the right
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class PanningController(nn.Module):
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
            >>> # Initialize controller
            >>> panner = StereoPanning(sample_rate=44100)
            >>> controller = PanningController(input_size=16)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = {'pan': controller(features)}
            >>> output = panner(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register the panning parameter.
        
        Sets up the pan parameter with range 0.0 (full left) to 1.0 (full right).
        """
        self.params = {
            'pan': EffectParam(min_val=0.0, max_val=1.0),
        } 
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the stereo panner.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 1, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'pan': Stereo position from left to right (0 to 1)
                - 'width': Stereo width/spread control (0 to 1)
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
        
        # get parameters 
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        pan = params['pan']
        bs, chs, seq_len = x.size()
        assert chs == 1, "Input tensor must have shape (bs, 1, seq_len)"
        
        theta = pan * (np.pi / 2)
        # compute gain coefficients
        left_gain = torch.sqrt(((np.pi / 2) - theta) * (2 / np.pi) * torch.cos(theta))
        right_gain = torch.sqrt(theta * (2 / np.pi) * torch.sin(theta))

        # make stereo
        # x = x.unsqueeze(1)
        x = x.repeat(1, 2, 1) # [bs, 2, seq_len]

        # apply panning
        left_gain = left_gain.view(bs, 1, 1)
        right_gain = right_gain.view(bs, 1, 1)
        gains = torch.cat((left_gain, right_gain), dim=1)
        x *= gains

        return x