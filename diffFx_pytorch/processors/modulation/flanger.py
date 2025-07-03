import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.utils import variable_delay
import math

# ref: https://www.audiokit.io/DunneAudioKit/documentation/dunneaudiokit/modulationeffects
# ref: https://ccrma.stanford.edu/~jos/pasp/Flanging.html

from torch import Tensor as T
import torch
import torch.nn as nn
from typing import Dict, Union

class Flanger(ProcessorsBase):
    """Differentiable implementation of a flanger audio effect.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    
    This processor implements a modulated delay line to create the flanger effect,
    using a low-frequency oscillator (LFO) to modulate a very short delay time. 
    The implementation creates the characteristic "swooshing" sound through 
    phase cancellation and reinforcement.

    Processing Chain:
        1. Generate LFO for delay modulation
        2. Calculate delay phases
        3. Apply variable delay
        4. Mix with original signal

    The transfer function is:

    .. math::

        y(t) = mix * x(t - d(t)) + (1 - mix) * x(t)
        
        d(t) = depth * sin(2πf_rt) + delay_{base}

    where coefficients are functions of:
        - x(t): Input signal
        - f_r: LFO rate in Hz
        - depth: Modulation depth
        - delay_base: Base delay time
        - mix: Wet/dry balance
    """
    def _register_default_parameters(self):
        """Register default parameters for the flanger effect."""
        self.params = {
            'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
            'rate': EffectParam(min_val=0.1, max_val=2.0),         # More musical range
            'depth': EffectParam(min_val=0.0, max_val=1.0),        # Full range
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def __init__(self, sample_rate=44100, param_range=None):
        super().__init__(sample_rate, param_range)
        self.sample_rate = sample_rate
        self._register_default_parameters()
        
    def process(self, 
        x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        """Process input signal through the flanger effect.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                    - 'delay_ms': Base delay time (0 to 1)
                    - 'rate': LFO frequency (0 to 1)
                    - 'depth': Modulation intensity (0 to 1)
                    - 'mix': Wet/dry balance (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify flanger parameters as:
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
        batch_size, n_ch, n_samples = x.shape
        device = x.device
        
        delay_ms = params['delay_ms'].view(-1, 1, 1)    # (batch, 1, 1)
        rate = params['rate'].view(-1, 1, 1)            # (batch, 1, 1)
        depth = params['depth'].view(-1, 1, 1)          # (batch, 1, 1)
        mix = params['mix'].view(-1, 1, 1)              # (batch, 1, 1)
        
        max_delay_samples = max(1, int(torch.max(delay_ms) * self.sample_rate / 1000.0))
        delay_center = delay_ms / 1000.0 * self.sample_rate # samples 
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        delay_lfo = torch.sin(2 * math.pi * rate * time.view(1, 1, -1)) 
        delay_value = delay_lfo * (depth * delay_center) + delay_center # 
        delay_phase = delay_value / max_delay_samples
        delay_phase = delay_phase.expand(-1, n_ch, -1)  
        delayed = variable_delay(delay_phase, x, buf_size=math.ceil(max_delay_samples))
        return mix * delayed + (1 - mix) * x

class StereoFlanger(ProcessorsBase):
    """Differentiable implementation of a stereo flanger effect with quadrature LFOs.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
        
    This processor implements a stereo flanger that uses quadrature (90° phase-shifted) 
    LFOs for the left and right channels, creating a wide stereo image through 
    independent modulation. The implementation provides smooth phase differences 
    between channels while maintaining the characteristic flanger sound.

    Processing Chain:
    1. Generate quadrature LFOs for stereo modulation
    2. Calculate independent channel delays
    3. Apply stereo variable delay
    4. Mix with original signal

    The transfer function for each channel is:

    .. math::

        y_L(t) = mix * x_L(t - d_L(t)) + (1 - mix) * x_L(t)
    
        y_R(t) = mix * x_R(t - d_R(t)) + (1 - mix) * x_R(t)
    
        d_L(t) = depth * sin(2πf_rt) + delay_{base}
    
        d_R(t) = depth * sin(2πf_rt + π/2) + delay_{base}

    where coefficients are functions of:
    - x_L, x_R: Left and right input signals
    - f_r: LFO rate in Hz
    - depth: Modulation depth
    - delay_base: Base delay time
    - mix: Wet/dry balance
    """
    def _register_default_parameters(self):
        """Register default parameters for the stereo flanger effect."""
        self.params = {
            'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
            'rate': EffectParam(min_val=0.1, max_val=2.0),         # More musical range
            'depth': EffectParam(min_val=0.0, max_val=1.0),        # Full range
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def __init__(self, sample_rate=44100, param_range=None):
        super().__init__(sample_rate, param_range)
        self.sample_rate = sample_rate
        self._register_default_parameters()
        
    def process(self, 
        x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        """Process input signal through the stereo flanger effect.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 2, samples)
                Must be stereo input (2 channels).
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                    - 'delay_ms': Base delay time (0 to 1)
                    - 'rate': LFO frequency (0 to 1)
                    - 'depth': Modulation intensity (0 to 1)
                    - 'mix': Wet/dry balance (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify flanger parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor. Shape: (batch, 2, samples)
            
        Raises:
            AssertionError: If input is not stereo (2 channels)
        """
        # Get parameters
        check_params(norm_params, dsp_params)
        # Set proper configuration
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        batch_size, n_ch, n_samples = x.shape
        assert n_ch == 2, "Input tensor must have shape (bs, 2, seq_len)"
        device = x.device
        delay_ms = params['delay_ms'].view(-1, 1, 1)    # (batch, 1, 1)
        rate = params['rate'].view(-1, 1, 1)            # (batch, 1, 1)
        depth = params['depth'].view(-1, 1, 1)          # (batch, 1, 1)
        mix = params['mix'].view(-1, 1, 1)              # (batch, 1, 1)
        max_delay_samples = max(1, int(torch.max(delay_ms) * self.sample_rate / 1000.0))
        delay_center = delay_ms / 1000.0 * self.sample_rate # samples 
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        phase_left = 2 * math.pi * rate * time.view(1, 1, -1)
        phase_right = phase_left + math.pi/2  # 
        delay_lfo_left = torch.sin(phase_left)   # LFO
        delay_lfo_right = torch.sin(phase_right) # LFO
        delay_lfo = torch.cat([delay_lfo_left, delay_lfo_right], dim=1)  # (batch, 2, samples)
        delay_value = delay_lfo * (depth * delay_center) + delay_center
        delay_phase = delay_value / max_delay_samples
        delayed = variable_delay(delay_phase, x, buf_size=math.ceil(max_delay_samples))
        return mix * delayed + (1 - mix) * x
     
class FeedbackFlanger(ProcessorsBase):
    """Differentiable implementation of a feedback flanger effect.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    
    This processor implements a flanger with feedback path, allowing the delayed signal
    to be fed back into the input. The feedback creates resonant peaks in the frequency
    response, resulting in a more pronounced and characteristically "metallic" flanger sound.

    Processing Chain:
    1. Generate LFO for delay modulation
    2. Sum input with feedback signal
    3. Apply modulated delay
    4. Feed delayed signal back
    5. Mix with original signal

    The transfer function with feedback is:

    .. math::

        y(t) = mix * (x(t) + fb * y(t - d(t))) + (1 - mix) * x(t)
    
        d(t) = depth * sin(2πf_rt) + delay_{base}

    where coefficients are functions of:
    - x(t): Input signal
    - f_r: LFO rate in Hz
    - depth: Modulation depth
    - delay_base: Base delay time
    - fb: Feedback amount
    - mix: Wet/dry balance
    """
    def _register_default_parameters(self):
        """Register default parameters for the feedback flanger effect"""
        self.params = {
            'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
            'rate': EffectParam(min_val=0.1, max_val=10.0),         # More musical range
            'depth': EffectParam(min_val=0.0, max_val=0.25),        # Full range
            'feedback': EffectParam(min_val=0.0, max_val=0.7),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def __init__(self, sample_rate=44100, param_range=None):
        super().__init__(sample_rate, param_range)
        self.sample_rate = sample_rate
        self._register_default_parameters()
        
    def process(self, 
        x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        """Process input signal through the feedback flanger effect.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                    - 'delay_ms': Base delay time (0 to 1)
                    - 'rate': LFO frequency (0 to 1)
                    - 'depth': Modulation intensity (0 to 1)
                    - 'feedback': Feedback amount (0 to 1)
                    - 'mix': Wet/dry balance (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify flanger parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
        """
        # Get parameters
        check_params(norm_params, dsp_params)
        # Set proper configuration
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        batch_size, n_ch, n_samples = x.shape
        device = x.device
        delay_ms = params['delay_ms'].view(-1, 1, 1)    # (batch, 1, 1)
        rate = params['rate'].view(-1, 1, 1)            # (batch, 1, 1)
        depth = params['depth'].view(-1, 1, 1)          # (batch, 1, 1)
        mix = params['mix'].view(-1, 1, 1)              # (batch, 1, 1)
        feedback = params['feedback'].view(-1, 1, 1)
        
        max_delay_samples = max(1, int(torch.max(delay_ms) * self.sample_rate / 1000.0))
        delay_center = delay_ms / 1000.0 * self.sample_rate # samples 
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        phase = 2 * math.pi * rate * time.view(1, 1, -1) 
        delay_lfo = torch.sin(phase) 
        delay_value = delay_lfo * (depth * delay_center) + delay_center # 
        delay_phase = delay_value / max_delay_samples
        delay_phase = delay_phase.expand(-1, n_ch, -1) 
        delayed = x
        delayed = variable_delay(delay_phase, x + feedback * delayed, buf_size=math.ceil(max_delay_samples))
        return mix * delayed + (1 - mix) * x
    
    
