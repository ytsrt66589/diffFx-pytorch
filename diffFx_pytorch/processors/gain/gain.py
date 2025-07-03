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
    """
    def __init__(self, sample_rate: int, param_range: Dict[str, EffectParam] = None):
        super().__init__(sample_rate, param_range)

    def _register_default_parameters(self):
        """Register gain parameter (-12.0 to 12.0 dB)."""
        self.params = {
            'gain_db': EffectParam(min_val=-12.0, max_val=12.0)
        }
    
    def process(
        self, 
        x: torch.Tensor, 
        nn_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, Union[float, torch.Tensor]], None] = None
    ):
        """Process input signal through the gain stage.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            nn_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify gain_db as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size and moved to input device if necessary. If provided, norm_params must be None.
        
        Returns:
            torch.Tensor: Processed audio tensor of same shape as input. Shape: (batch, channels, samples)
        """
        check_params(nn_params, dsp_params)
        
        if nn_params is not None:
            params = self.map_parameters(nn_params)
        else:
            params = dsp_params
        
        gain_db = params['gain_db']
        gain = 10 ** (gain_db / 20.0)
        gain = gain.unsqueeze(-1).unsqueeze(-1)
        
        return x * gain
    
