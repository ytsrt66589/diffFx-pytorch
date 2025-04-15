import torch 
from typing import Dict, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params


class Gain(ProcessorsBase):
    """Differentiable implementation of a simple gain/volume control.
    
    This processor implements a basic gain stage that applies amplitude scaling 
    to the input signal. The gain is specified in decibels and converted to 
    linear scaling internally.

    The gain calculation follows:

    .. math::

        y[n] = x[n] * 10^{gain_{dB}/20}

    where:
        - x[n] is the input signal
        - gain_dB is the gain in decibels
        - Division by 20 converts dB to amplitude ratio

    Args:
        sample_rate (int): Audio sample rate in Hz 
        param_range (Dict[str, EffectParam]): Parameter range for the gain control
    
    Parameters Details:
        gain_db: Gain control in decibels
            - Default Range: -12.0 to 12.0 dB
            - Logarithmic control for natural volume scaling

    Examples:
        Basic DSP Usage:
            >>> # Create a gain stage
            >>> gain = Gain(sample_rate=44100)
            >>> # Process audio
            >>> output = gain(input_audio, dsp_params={
            ...     'gain_db': 6.0  # Boost by 6 dB (2x amplitude)
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class GainController(nn.Module):
            ...     def __init__(self, input_size):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, 1),  # Single gain parameter
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Process with features
            >>> controller = GainController(input_size=16)
            >>> features = torch.randn(batch_size, 16)
            >>> norm_params = controller(features)
            >>> output = gain(input_audio, norm_params=norm_params)
    """
    def __init__(self, sample_rate: int, param_range: Dict[str, EffectParam] = None):
        super().__init__(sample_rate, param_range)

    def _register_default_parameters(self):
        """Register gain parameter (-12.0 to 12.0 dB)."""
        self.params = {
            'gain_db': EffectParam(min_val=-12.0, max_val=12.0)
        }
    
    def process(self, 
        x: torch.Tensor, 
        norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, Union[float, torch.Tensor]], None] = None
    ):
        """Process input signal through the gain stage.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify gain_db as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.
        
        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
        """
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        # Get gain parameter and convert to linear scale
        gain_db = params['gain_db']
        gain = 10 ** (gain_db / 20.0)
        
        # Reshape gain for broadcasting
        gain = gain.unsqueeze(-1).unsqueeze(-1)
        
        # Apply gain
        return x * gain
    
