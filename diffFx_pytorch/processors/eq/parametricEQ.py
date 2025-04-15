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

    Args:
        sample_rate (int): Audio sample rate in Hz
        param_range (Dict[str, EffectParam], optional): Parameter ranges.
        num_peak_filters (int): Number of independent peak filters. Defaults to 3.

    Parameters Details:
        Low Shelf Section:
            - low_shelf_gain_db: Gain for low frequencies
                - Range: -12.0 to 12.0 dB
                - Controls bass boost/cut
            - low_shelf_frequency: Corner frequency
                - Range: 20.0 to 500.0 Hz
                - Controls bass transition point
            - low_shelf_q_factor: Slope control
                - Range: 0.1 to 1.0
                - Controls bass shelf slope

        Peak Filter Sections (for each peak filter i):
            - peak_i_gain_db: Gain for the band
                - Range: -12.0 to 12.0 dB
                - Controls midrange boost/cut
            - peak_i_frequency: Center frequency
                - Range: 20.0 to 20000.0 Hz
                - Controls midrange center point
            - peak_i_q_factor: Bandwidth control
                - Range: 0.1 to 10.0
                - Controls midrange bandwidth

        High Shelf Section:
            - high_shelf_gain_db: Gain for high frequencies
                - Range: -12.0 to 12.0 dB
                - Controls treble boost/cut
            - high_shelf_frequency: Corner frequency
                - Range: 5000.0 to 20000.0 Hz
                - Controls treble transition point
            - high_shelf_q_factor: Slope control
                - Range: 0.1 to 1.0
                - Controls treble shelf slope

    Note:
        The equalizer consists of three main sections:
            - Low shelf filter for bass control
            - Multiple peak filters for midrange control
            - High shelf filter for treble control
        Each section provides independent control over gain, frequency, and Q factor,
        allowing for precise frequency response shaping.

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their ranges
            - Ensure your network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()

    Examples:
        Basic DSP Usage:
            >>> # Create a parametric EQ with 3 peak filters
            >>> eq = ParametricEqualizer(
            ...     sample_rate=44100,
            ...     num_peak_filters=3
            ... )
            >>> # Process audio with dsp parameters
            >>> params = {
            ...     'low_shelf_gain_db': 6.0,
            ...     'low_shelf_frequency': 100.0,
            ...     'low_shelf_q_factor': 0.7,
            ...     'peak_1_gain_db': -3.0,
            ...     'peak_1_frequency': 1000.0,
            ...     'peak_1_q_factor': 1.4,
            ...     # ... additional peak filter parameters ...
            ...     'high_shelf_gain_db': -2.0,
            ...     'high_shelf_frequency': 8000.0,
            ...     'high_shelf_q_factor': 0.7
            ... }
            >>> output = eq(input_audio, dsp_params=params)

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class ParametricEQController(nn.Module):
            ...     def __init__(self, input_size, num_parameters):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, num_parameters),
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Initialize controller
            >>> eq = ParametricEqualizer(num_peak_filters=3)
            >>> num_params = eq.count_num_parameters()  # 15 parameters for 3 peak filters
            >>> controller = ParametricEQController(input_size=16, num_parameters=num_params)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = controller(features)
            >>> output = eq(input_audio, norm_params=norm_params)
    """
    def __init__(self, sample_rate, param_range = None, num_peak_filters=3):
        self.num_peak_filters = num_peak_filters
        super().__init__(sample_rate, param_range)

        # Create peak filters
        self.peak_filters = [
            BiquadFilter(
                sample_rate=self.sample_rate,
                filter_type='pk'
            ) for _ in range(num_peak_filters)
        ]
        
        # Create low shelf and high shelf filters
        self.low_shelf_filter = BiquadFilter(
            sample_rate=self.sample_rate,
            filter_type='ls'
        )
        
        self.high_shelf_filter = BiquadFilter(
            sample_rate=self.sample_rate,
            filter_type='hs'
        )
        
    def _register_default_parameters(self):
        """Register parameters for all filter sections.
        
        Sets up parameter ranges for:
            - Low shelf filter (gain, frequency, Q)
            - Multiple peak filters (gain, frequency, Q for each)
            - High shelf filter (gain, frequency, Q)
            
        Each parameter is registered with appropriate min/max values for its function.
        """
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
            torch.Tensor: Processed audio tensor of same shape as input
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
    
    
