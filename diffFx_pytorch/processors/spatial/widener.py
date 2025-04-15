import torch 
import numpy as np 
from typing import Dict, Union
from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..core.midside import * 

class StereoWidener(ProcessorsBase):
    """Differentiable implementation of mid-side stereo width control.
    
    This processor implements stereo width adjustment using mid-side (M/S) processing,
    allowing continuous control from mono to enhanced stereo width. It operates by
    converting the input to M/S representation, scaling the side signal, and converting
    back to left-right stereo.

    The width control is implemented using the following process:
    
    .. math::

        M_{out} = M_{in} * 2(1 - width)
        
        S_{out} = S_{in} * 2(width)

    where:
        - M is the mid (mono) signal: (L + R) / √2
        - S is the side (difference) signal: (L - R) / √2
        - width is the stereo width control parameter
        - Scaling ensures energy preservation across width settings

    Processing Chain:
        1. Convert L/R to M/S representation
        2. Scale mid and side signals based on width
        3. Convert back to L/R representation

    Args:
        sample_rate (int): Audio sample rate in Hz

    Parameters Details:
        width: Stereo width control
            - 0.0: Mono (side signal removed)
            - 0.5: Original stereo (no change)
            - 1.0: Enhanced stereo (doubled side signal)
            - Continuously variable between these points
            - Maintains constant total energy

    Note:
        - Input must be stereo (two channels)
        - Uses energy-preserving M/S conversion matrices
        - Width control affects the ratio of mid to side signal
        - Extreme width settings may cause phase issues
        - Mono compatibility is maintained across all settings

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameter will be automatically mapped to width range
            - Ensure your network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()

    Examples:
        Basic DSP Usage:
            >>> # Create a stereo widener
            >>> widener = StereoWidener(sample_rate=44100)
            >>> # Process stereo audio with direct width control
            >>> output = widener(input_audio, dsp_params={
            ...     'width': 0.75  # Enhance stereo width by 50%
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class WidthController(nn.Module):
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
            >>> widener = StereoWidener(sample_rate=44100)
            >>> controller = WidthController(input_size=16)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = {'width': controller(features)}
            >>> output = widener(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register the width parameter.
        
        Sets up the width parameter with range:
            - 0.0: Mono (collapse to center)
            - 0.5: No change (original stereo)
            - 1.0: Enhanced stereo (maximum width)
        """
        # 0.0 -> mono 0.5 -> no change 1.0 -> stereo
        self.params = {
            'width': EffectParam(min_val=0.0, max_val=1.0),
        } 
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the stereo widener.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 2, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'width': Stereo width control (0 to 1)
                    0.0: Mono/centered
                    0.5: Original stereo width
                    1.0: Maximum width
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify widener parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor. Shape: (batch, 2, samples)
            
        Raises:
            AssertionError: If input is not stereo (two channels)
        """
        check_params(norm_params, dsp_params)
        
        # get parameters 
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        width = params['width']
        bs, chs, seq_len = x.size()
        assert chs == 2, "Input tensor must have shape (bs, 2, seq_len)"
        
        x_ms = lr_to_ms(x, mult=1/np.sqrt(2)) 
        
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
        x_lr = ms_to_lr(x_ms, mult=1/np.sqrt(2))
        
        return x_lr