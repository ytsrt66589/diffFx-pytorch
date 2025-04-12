import math
import torch
import torch.fft

from ..core.iir import IIRFilter
from ..core.utils import freq_to_normalized
from typing import Dict, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params

PI = math.pi
TWO_PI = 2 * math.pi
HALF_PI = math.pi / 2
TWOR_SCALE = 1 / math.log(2)
ALPHA_SCALE = 1 / 2

class BiquadFilter(ProcessorsBase):
    """Differentiable implementation of a second-order IIR (biquad) filter.
    
    This processor implements various types of biquad filters based on the Audio EQ Cookbook
    formulas. It supports multiple filter types including lowpass, highpass, bandpass, 
    bandstop, allpass, peaking, and shelving filters. The filter is implemented using the 
    following transfer function:

    .. math::

        H(z) = \\frac{b_0 + b_1z^{-1} + b_2z^{-2}}{a_0 + a_1z^{-1} + a_2z^{-2}}

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        param_ranges (Dict[str, Tuple[str, EffectParam]]): Optional parameter definitions to override defaults.
        filter_type (str): Type of filter to implement (see Filter Types below).
        **kwargs: Additional arguments passed to IIRFilter

    Parameters Details:
        frequency: Center/cutoff frequency
            - Range: 20.0 to 20000.0 Hz
            - Controls filter's frequency response
            - Meaning depends on filter type
            
        q_factor: Quality factor
            - Range: 0.1 to 10.0
            - Controls bandwidth/resonance
            - Higher values = narrower/sharper response
            
        gain_db: Gain for peak/shelf filters
            - Range: -12.0 to 12.0 dB
            - Only used by peak and shelf types
            - Controls amount of boost/cut

    Filter Types:
        Low-pass (LP):
            - Passes frequencies below cutoff
            - -12 dB/octave slope
            
        High-pass (HP):
            - Passes frequencies above cutoff
            - -12 dB/octave slope
            
        Band-pass (BP):
            - Passes frequencies around center
            - Constant skirt gain
            
        Band-stop (BS):
            - Rejects frequencies around center
            - Also known as notch filter
            
        All-pass (AP):
            - Maintains magnitude response
            - Adjusts phase response
            
        Peak (PK):
            - Boosts/cuts around center
            
        Low-shelf (LS):
            - Boosts/cuts below frequency
            
        High-shelf (HS):
            - Boosts/cuts above frequency

    Examples:
        Basic DSP Usage:
            >>> # Create a lowpass filter
            >>> lpf = BiquadFilter(
            ...     sample_rate=44100,
            ...     filter_type='LP'
            ... )
            >>> # Process audio
            >>> output = lpf(input_audio, dsp_params={
            ...     'frequency': 1000.0,  # 1 kHz cutoff
            ...     'q_factor': 0.707,    # Butterworth response
            ...     'gain_db': 0.0        # Not used for LP
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class FilterController(nn.Module):
            ...     def __init__(self, input_size):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, 3),  # 3 parameters
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
    """
    def _register_default_parameters(self):
        """Register filter parameters with ranges.
        
        Sets up three parameters:
            - frequency: Center/cutoff in Hz (20.0 to 20000.0)
            - q_factor: Quality factor (0.1 to 10.0)
            - gain_db: Gain in dB (-12.0 to 12.0)
        """
        self.params = {
            'frequency': EffectParam(min_val=20.0, max_val=20000.0),  # Hz
            'q_factor': EffectParam(min_val=0.1, max_val=10.0),  # standard Q
            'gain_db': EffectParam(min_val=-12.0, max_val=12.0),  # for peak/shelf
        }
        
    def __init__(self, sample_rate=44100, param_range=None, filter_type='LP', **kwargs):
        super().__init__(sample_rate, param_range)
        self.biquad = IIRFilter(order=2, **kwargs)
        self.filter_type = self._map_filter_type_2_num(filter_type)
        
    def _map_filter_type_2_num(self, filter_type: str) -> int:
        """Map filter type string to internal numeric representation.

        Args:
            filter_type (str): Filter type identifier (see class docstring for supported types)
            
        Returns:
            int: Internal filter type number
            
        Raises:
            ValueError: If filter type is not recognized
        """
        filter_map = {
            'lowpass': 0,  'lp': 0,
            'highpass': 1, 'hp': 1,
            'bandpass': 2, 'bp': 2,
            'bandstop': 3, 'bs': 3, 'notch': 3,
            'allpass': 4,  'ap': 4,
            'peak': 5,     'pk': 5,
            'lowshelf': 6, 'ls': 6,
            'highshelf': 7,'hs': 7
        }
        
        filter_type = filter_type.lower().replace(' ', '')
        
        if filter_type not in filter_map:
            raise ValueError(
                f"Unknown filter type: {filter_type}. "
                f"Must be one of {list(set(k for k in filter_map.keys() if len(k) > 2))}"
            )
            
        return filter_map[filter_type]
        
    def process(
        self, 
        x: torch.Tensor, 
        norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ):
        """Process input signal through the biquad filter.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, torch.Tensor], optional): Direct DSP parameters.
                If provided, norm_params must be None.
                
        Returns:
            torch.Tensor: Filtered audio tensor of same shape as input
        """
        check_params(norm_params, dsp_params)
        
        # get parameters
        if norm_params is not None:
            
            params = self.map_parameters(norm_params) # map normalized parameters to DSP values
            frequency = params['frequency']
            q_factor = params['q_factor']
            gain_db = params['gain_db']
        else:
            frequency = dsp_params['frequency']
            q_factor = dsp_params['q_factor']
            gain_db = dsp_params['gain_db']
        
        filter_type = self.filter_type
        
        # convert to filter parameters
        w0 = freq_to_normalized(frequency, self.sample_rate)
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        alpha = sin_w0 / (2 * q_factor)
        A = torch.pow(10, gain_db / 40.0)  # for peak/shelf filters
        
        A_p_1 = A + 1
        A_m_1 = A - 1
        A_p_1_cos_w0 = A_p_1 * cos_w0
        A_m_1_cos_w0 = A_m_1 * cos_w0
        A_sqrt = torch.sqrt(A)
        two_A_sqrt_alpha = 2 * A_sqrt * alpha
        
        # calculate coefficients based on filter type
        if filter_type == 0:  # Low Pass # valid 
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = (1 - cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif filter_type == 1:  # High Pass # valid 
            b0 = (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 = (1 + cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif filter_type == 2:  # Band Pass # valid
            b0 = alpha
            b1 = torch.zeros(b0.shape).to(b0.device)#0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif filter_type == 3:  # Band Stop/Reject # valid
            # b0 = 1
            b1 = -2 * cos_w0
            b2 = torch.ones(b1.shape).to(b1.device)
            b0 = torch.ones(b1.shape).to(b1.device)
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif filter_type == 4:  # All Pass # valid
            b0 = 1 - alpha
            b1 = -2 * cos_w0
            b2 = 1 + alpha
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif filter_type == 5:  # Peak # valid
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
            
        elif filter_type == 6:  # Low Shelf # valid
            b0 = A * (A_p_1 - A_m_1_cos_w0 + two_A_sqrt_alpha)
            b1 = 2 * A * (A_m_1 - A_p_1_cos_w0)
            b2 = A * (A_p_1 - A_m_1_cos_w0 - two_A_sqrt_alpha)
            a0 = A_p_1 + A_m_1_cos_w0 + two_A_sqrt_alpha
            a1 = -2 * (A_m_1 + A_p_1_cos_w0)
            a2 = A_p_1 + A_m_1_cos_w0 - two_A_sqrt_alpha
            
        else:  # High Shelf # valid
            b0 = A * (A_p_1 + A_m_1_cos_w0 + two_A_sqrt_alpha)
            b1 = -2 * A * (A_m_1 + A_p_1_cos_w0)
            b2 = A * (A_p_1 + A_m_1_cos_w0 - two_A_sqrt_alpha)
            a0 = A_p_1 - A_m_1_cos_w0 + two_A_sqrt_alpha
            a1 = 2 * (A_m_1 - A_p_1_cos_w0)
            a2 = A_p_1 - A_m_1_cos_w0 - two_A_sqrt_alpha
        # print(b0)
        
        # stack coefficients
        
        Bs = torch.stack([b0, b1, b2], -1).unsqueeze(1)
        As = torch.stack([a0, a1, a2], -1).unsqueeze(1)
        
        # apply filter
        x_filtered = self.biquad(x, Bs, As)
        return x_filtered

