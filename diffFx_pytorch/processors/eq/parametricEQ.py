import torch 
from typing import Dict, Union

from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..filters import BiquadFilter


# Parametric Equalizer 
class ParametricEqualizer(ProcessorsBase):
    """Differentiable implementation of a parametric equalizer.
    
    Implementation is based on following book and papers: 

    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] Lee, Sungho, et al. "GRAFX: an open-source library for audio processing graphs in PyTorch." 
            arXiv preprint arXiv:2408.03204 (2024).
    
    This processor implements a versatile parametric equalizer combining multiple peak filters
    with high and low shelf filters. Each filter section provides independent control over
    gain, frequency, and bandwidth (Q factor), offering precise frequency response shaping
    capabilities.

    Each filter section uses a second-order IIR (biquad) implementation with transfer function:

    .. math::

        H(z) = \\frac{b_0 + b_1z^{-1} + b_2z^{-2}}{1 + a_1z^{-1} + a_2z^{-2}}

    where coefficients are computed based on:
        - Filter type (peak, low shelf, or high shelf)
        - Center frequency
        - Q factor (bandwidth/slope)
        - Gain setting
    """
    def __init__(self, sample_rate, param_range = None, num_peak_filters=3):
        self.num_peak_filters = num_peak_filters
        super().__init__(sample_rate, param_range)

        self.peak_filters = [
            BiquadFilter(
                sample_rate=self.sample_rate,
                filter_type='pk'
            ) for _ in range(num_peak_filters)
        ]
        self.low_shelf_filter = BiquadFilter(
            sample_rate=self.sample_rate,
            filter_type='ls'
        )
        self.high_shelf_filter = BiquadFilter(
            sample_rate=self.sample_rate,
            filter_type='hs'
        )
        
    def _register_default_parameters(self):
        """Register parameters for all filter sections."""
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
            
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the parametric equalizer.

        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Dictionary with keys for each parameter:
                - Low shelf: 'low_shelf_gain_db', 'low_shelf_frequency', 'low_shelf_q_factor'
                - Peak filters: 'peak_X_gain_db', 'peak_X_frequency', 'peak_X_q_factor' for X in range(1, num_peak_filters+1)
                - High shelf: 'high_shelf_gain_db', 'high_shelf_frequency', 'high_shelf_q_factor'
                Each value should be a tensor of shape (batch_size,)
                Values will be mapped to their respective ranges 
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input. Shape: (batch, channels, samples)
        """
        check_params(norm_params, dsp_params)
        
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
                    'gain_db': dsp_params[f'{peak_name}_gain_db'],
                    'frequency': dsp_params[f'{peak_name}_frequency'],
                    'q_factor': dsp_params[f'{peak_name}_q_factor']
                }
                x_processed = self.peak_filters[i](x_processed, None, peak_params)
            
            # Apply high shelf filter
            high_shelf_params = {
                'gain_db': dsp_params['high_shelf_gain_db'],
                'frequency': dsp_params['high_shelf_frequency'],
                'q_factor': dsp_params['high_shelf_q_factor']
            }
            x_processed = self.high_shelf_filter(x_processed, None, high_shelf_params)
        
        return x_processed
    
    
