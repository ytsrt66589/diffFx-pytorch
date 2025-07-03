import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.utils import variable_delay
import math

class Chorus(ProcessorsBase):
    """Differentiable implementation of a chorus audio effect.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] https://github.com/hyakuchiki/diffsynth/blob/master/diffsynth/modules/delay.py
    
    This processor implements a modulated delay line to create the chorus effect,
    generating multiple detuned copies of the input signal using LFO-controlled
    delay modulation. 

    Processing Chain:
        1. Generate LFO for delay modulation
        2. Calculate delay phases
        3. Apply variable delay
        4. Mix with original signal

    The transfer function is:

    .. math::

        y(t) = mix * (x(t) + LFO_{delay}(t)) + (1 - mix) * x(t)

    where coefficients are functions of:
        - x(t): Input signal
        - LFO_{delay}(t): Delay modulated by sine wave
        - mix: Wet/dry balance parameter
    """
    def _register_default_parameters(self):
        """Register default parameters for the chorus effect.""" 
        self.params = {
            'delay_ms': EffectParam(min_val=10.0, max_val=25.0),   
            'rate': EffectParam(min_val=0.1, max_val=10.0),         
            'depth': EffectParam(min_val=0.0, max_val=0.25),        
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def __init__(self, sample_rate=44100, param_range=None):
        super().__init__(sample_rate, param_range)
        self._register_default_parameters()
        
    def process(
        self, 
        x: torch.Tensor, 
        norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        """Process input signal through the chorus effect.
    
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
                Can specify chorus parameters as:
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
        
        delay_ms = params['delay_ms'].view(-1, 1, 1)    
        rate = params['rate'].view(-1, 1, 1)            
        depth = params['depth'].view(-1, 1, 1)          
        mix = params['mix'].view(-1, 1, 1)              
        
        max_delay_samples = max(
            1, 
            int(torch.max(delay_ms) * self.sample_rate / 1000.0)
        )
        delay_center = delay_ms / 1000.0 * self.sample_rate
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        phase = 2 * math.pi * rate * time.view(1, 1, -1) 
        delay_lfo = torch.sin(phase) 
        delay_value = delay_lfo * (depth * delay_center) + delay_center
        delay_phase = delay_value / max_delay_samples
        delay_phase = delay_phase.expand(-1, n_ch, -1)
        delayed = variable_delay(delay_phase, x, buf_size=math.ceil(max_delay_samples))
        return mix * delayed + (1 - mix) * x
    
    
class MultiVoiceChorus(ProcessorsBase):
    """Differentiable implementation of a multi-voice chorus effect with independent voice control.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    
    This processor implements a chorus effect with multiple independently controlled voices,
    each featuring phase-shifted modulation and individual gain control. The implementation 
    uses multiple LFO-modulated delay lines with evenly distributed phase offsets to create
    rich ensemble effects.

    The transfer function for the multi-voice chorus is:

    .. math::

        y(t) = (1-mix)x(t) + mix\\sum_{i=0}^{N-1} g_i x(t - d_i(t))
        
        d_i(t) = depth * sin(2πf_rt + \\frac{2πi}{N}) + delay_{base}

    where coefficients are functions of:
        - N: Number of chorus voices
        - g_i: Gain of voice i
        - f_r: LFO rate in Hz
        - depth: Modulation depth
        - delay_base: Base delay time
    """
    def _register_default_parameters(self):
        """Register default parameters for the multi-voice chorus."""
        self.params = {
            'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    
            'rate': EffectParam(min_val=0.1, max_val=10.0),        
            'depth': EffectParam(min_val=0.0, max_val=0.25),       
            'mix': EffectParam(min_val=0.0, max_val=1.0),          
        }
        
        for i in range(self.num_voices):
            self.params[f'g{i}'] = EffectParam(min_val=0.0, max_val=1.0)
        
    def __init__(self, sample_rate=44100, param_range=None, num_voices=2):
        self.num_voices = num_voices
        super().__init__(sample_rate, param_range)
        self._register_default_parameters()
        
    def process(
        self, 
        x: torch.Tensor, 
        norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        """Process input signal through the multi-voice chorus.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                    - 'delay_ms': Base delay time (0 to 1)
                    - 'rate': LFO frequency (0 to 1)
                    - 'depth': Modulation intensity (0 to 1)
                    - 'mix': Wet/dry balance (0 to 1)
                    - 'g0' to 'g{num_voices-1}': Voice gains (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify chorus parameters as:
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
        # Set proper configuration
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        batch_size, n_ch, n_samples = x.shape
        device = x.device
        
        delay_ms = params['delay_ms'].view(-1, 1, 1)
        rate = params['rate'].view(-1, 1, 1)
        depth = params['depth'].view(-1, 1, 1)
        mix = params['mix'].view(-1, 1, 1)
        
        max_delay_samples = max(
            1, 
            int(torch.max(delay_ms) * self.sample_rate / 1000.0)
        )
        delay_center = delay_ms / 1000.0 * self.sample_rate
        
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        
        mixed_output = torch.zeros_like(x)
        
        for i in range(self.num_voices):
            gain = params[f'g{i}'].view(-1, 1, 1)
            
            phase = 2 * math.pi * rate * time.view(1, 1, -1) + (i * 2 * math.pi / self.num_voices)
            delay_lfo = torch.sin(phase)
            
            delay_value = delay_lfo * (depth * delay_center) + delay_center
            delay_phase = delay_value / max_delay_samples
            
            delay_phase = delay_phase.expand(-1, n_ch, -1)
            
            delayed = variable_delay(delay_phase, x, buf_size=math.ceil(max_delay_samples))
            mixed_output += gain * delayed
        
        y = mix * mixed_output + (1 - mix) * x
        
        return y
    
    
class StereoChorus(ProcessorsBase):
    """Differentiable implementation of a stereo chorus effect with configurable voices.
    
        Implementation is based on: 
        
        ..  [1] Reiss, Joshua D., and Andrew McPherson. 
                Audio effects: theory, implementation and application. CRC Press, 2014.
        
        This processor implements a stereo chorus effect with multiple independently controlled voices,
        each featuring phase-shifted modulation, individual gain control, and stereo panning. The
        implementation combines multi-voice modulated delays with constant-power stereo positioning
        to create rich, spatially distributed chorus effects.

        Processing Chain:
        1. Generate phase-offset LFOs for each voice
        2. Calculate individual voice delays
        3. Apply stereo panning per voice
        4. Mix delayed and panned voices
        5. Combine with dry signal

        The transfer function for each voice i is:

        .. math::

            y_L(t) = mix * \\sum_{i=0}^{N-1} g_i\\sqrt{\\frac{1-pan_i}{2}}x(t - d_i(t))
        
            y_R(t) = mix * \\sum_{i=0}^{N-1} g_i\\sqrt{\\frac{1+pan_i}{2}}x(t - d_i(t))
        
            d_i(t) = depth * sin(2πf_rt + \\frac{πi}{2}) + delay_{base}

        where coefficients are functions of:
        - N: Number of chorus voices
        - g_i: Gain of voice i
        - pan_i: Stereo position of voice i (-1 to 1)
        - f_r: LFO rate in Hz
        - depth: Modulation depth
        - delay_base: Base delay time
    """
    def _register_default_parameters(self):
        """Register default parameters for the stereo chorus."""
        self.params = {
            'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    
            'rate': EffectParam(min_val=0.1, max_val=10.0),        
            'depth': EffectParam(min_val=0.0, max_val=0.25),       
            'mix': EffectParam(min_val=0.0, max_val=1.0),          
        }
        
        for i in range(self.num_voices):
            self.params[f'g{i}'] = EffectParam(min_val=0.0, max_val=1.0)
            self.params[f'pan{i}'] = EffectParam(min_val=-1.0, max_val=1.0)
        
    def __init__(self, sample_rate=44100, param_range=None, num_voices=2):
        """Initialize the stereo chorus processor."""
        self.num_voices = num_voices
        super().__init__(sample_rate, param_range)
        self._register_default_parameters()
        
    def process(
        self, 
        x: torch.Tensor, 
        norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        """Process input signal through the stereo chorus.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
                Accepts both mono (channels=1) and stereo (channels=2) input.
                Mono input will be automatically converted to stereo.
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                    - 'delay_ms': Base delay time (0 to 1)
                    - 'rate': LFO frequency (0 to 1)
                    - 'depth': Modulation intensity (0 to 1)
                    - 'mix': Wet/dry balance (0 to 1)
                    For each voice i:
                        - f'g{i}': Voice gain (0 to 1)
                        - f'pan{i}': Voice position (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify chorus parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor. Shape: (batch, 2, samples)
        """
        check_params(norm_params, dsp_params)
        # Set proper configuration
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
            
        batch_size, n_ch, n_samples = x.shape
        device = x.device
        
        if n_ch == 1:
            x = x.repeat(1, 2, 1)
            n_ch = 2
        
        delay_ms = params['delay_ms'].view(-1, 1, 1)
        rate = params['rate'].view(-1, 1, 1)
        depth = params['depth'].view(-1, 1, 1)
        mix = params['mix'].view(-1, 1, 1)
        
        max_delay_samples = max(
            1, 
            int(torch.max(delay_ms) * self.sample_rate / 1000.0)
        )
        delay_center = delay_ms / 1000.0 * self.sample_rate
        
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        
        mixed_output = torch.zeros_like(x)
        
        for i in range(self.num_voices):
            gain = params[f'g{i}'].view(-1, 1, 1)
            pan = params[f'pan{i}'].view(-1, 1, 1)
            
            left_gain = gain * torch.sqrt(torch.clamp((1 - pan) / 2, 0, 1))
            right_gain = gain * torch.sqrt(torch.clamp((1 + pan) / 2, 0, 1))
            stereo_gains = torch.cat([left_gain, right_gain], dim=1)
            
            phase = 2 * math.pi * rate * time.view(1, 1, -1) + (i * math.pi / 2)
            delay_lfo = torch.sin(phase)
            
            delay_value = delay_lfo * (depth * delay_center) + delay_center
            delay_phase = delay_value / max_delay_samples
            
            delay_phase = delay_phase.expand(-1, n_ch, -1)
            
            delayed = variable_delay(delay_phase, x, buf_size=math.ceil(max_delay_samples))
            mixed_output += stereo_gains * delayed
        
        y = mix * mixed_output + (1 - mix) * x
        
        return y
    
    



