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

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.

    Attributes:
        sample_rate (int): Audio sample rate in Hz

    Parameters Details:
        delay_ms: Base delay time 
            - Range: 5.0 to 40.0 ms
            - Controls center delay time
            - Typical chorus uses 20-30ms
            
        rate: LFO modulation frequency
            - Range: 0.1 to 10.0 Hz
            - Controls modulation speed
            - Lower values create gentle chorusing
            - Higher values for vibrato effects
            
        depth: Modulation intensity
            - Range: 0.0 to 0.25
            - Controls amount of pitch variation
            - Affects richness of chorus effect
            
        mix: Wet/dry balance
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only chorused signal

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their ranges
            - Ensure your network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()

    Examples:
        Basic DSP Usage:
            >>> # Create a chorus effect
            >>> chorus = Chorus(
            ...     sample_rate=44100
            ... )
            >>> # Process with musical settings
            >>> output = chorus(input_audio, dsp_params={
            ...     'delay_ms': 20.0,  # 20ms base delay
            ...     'rate': 2.0,       # 2 Hz modulation
            ...     'depth': 0.15,     # Moderate intensity
            ...     'mix': 0.5         # Equal mix
            ... })

        Neural Network Control:
            >>> # Simple parameter prediction
            >>> class ChorusController(nn.Module):
            ...     def __init__(self, input_size, num_params):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, num_params),
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Initialize controller
            >>> chorus = Chorus(sample_rate=44100)
            >>> num_params = chorus.count_num_parameters()  # 4 parameters
            >>> controller = ChorusController(input_size=16, num_params=num_params)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = controller(features)
            >>> output = chorus(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register default parameters for the chorus effect.

        Sets up:
            delay_ms: Base delay time (5.0 to 40.0 ms)
            rate: LFO modulation rate (0.1 to 10.0 Hz)
            depth: Modulation intensity (0.0 to 0.25)
            mix: Wet/dry balance (0.0 to 1.0)
        """
        self.params = {
            'delay_ms': EffectParam(min_val=5.0, max_val=40.0),    # Increased range
            'rate': EffectParam(min_val=0.1, max_val=10.0),         # More musical range
            'depth': EffectParam(min_val=0.0, max_val=0.25),        # Full range
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
        # Set proper configuration
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        batch_size, n_ch, n_samples = x.shape
        device = x.device
        
        # Map parameters with correct shapes
        delay_ms = params['delay_ms'].view(-1, 1, 1)    # (batch, 1, 1)
        rate = params['rate'].view(-1, 1, 1)            # (batch, 1, 1)
        depth = params['depth'].view(-1, 1, 1)          # (batch, 1, 1)
        mix = params['mix'].view(-1, 1, 1)              # (batch, 1, 1)
        
        # Calculate maximum delay in samples
        max_delay_samples = max(
            1, 
            int(torch.max(delay_ms) * self.sample_rate / 1000.0)
        )
        delay_center = delay_ms / 1000.0 * self.sample_rate
        
        # Generate time base for LFO
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        
        # Generate LFO with batch dimension
        phase = 2 * math.pi * rate * time.view(1, 1, -1)  # (1, 1, n_samples)
        delay_lfo = torch.sin(phase)  # (batch, 1, n_samples)
        
        # Calculate delay values
        delay_value = delay_lfo * (depth * delay_center) + delay_center
        delay_phase = delay_value / max_delay_samples
        
        # Expand phase for all channels
        delay_phase = delay_phase.expand(-1, n_ch, -1)  # (batch, channel, n_samples)
        
        # Apply variable delay
        delayed = variable_delay(delay_phase, x, buf_size=math.ceil(max_delay_samples))
        
        # Mix dry and wet signals
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

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        num_voices (int): Number of chorus voices. Defaults to 2.

    Attributes:
        num_voices (int): Number of active chorus voices
        sample_rate (int): Audio sample rate in Hz

    Parameters Details:
        delay_ms: Base delay time
            - Range: 1.0 to 10.0 ms
            - Controls center delay time
            - Shorter than single chorus for tighter effect
            
        rate: LFO modulation frequency
            - Range: 0.1 to 10.0 Hz
            - Controls modulation speed
            - Affects movement of ensemble effect
            
        depth: Modulation intensity
            - Range: 0.0 to 0.25
            - Controls amount of detuning
            - Higher values create wider ensemble
            
        mix: Wet/dry balance
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only processed signal
            
        For each voice i:
            gi: Voice gain
                - Range: 0.0 to 1.0
                - Controls level of each voice
                - Allows voice balancing
        
    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to ranges
            - Ensure network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()

    Examples:
        Basic DSP Usage:
            >>> # Create a 3-voice chorus
            >>> chorus = MultiVoiceChorus(
            ...     sample_rate=44100,
            ...     num_voices=3
            ... )
            >>> # Process with musical settings
            >>> output = chorus(input_audio, dsp_params={
            ...     'delay_ms': 5.0,    # Base delay
            ...     'rate': 1.5,        # Modulation rate
            ...     'depth': 0.15,      # Moderate detuning
            ...     'mix': 0.7,         # Mostly wet
            ...     'g0': 1.0,          # Full level voice 1
            ...     'g1': 0.8,          # Reduced voice 2
            ...     'g2': 0.6           # Quiet voice 3
            ... })

        Neural Network Control:
            >>> # Simple parameter prediction
            >>> class MultiChorusController(nn.Module):
            ...     def __init__(self, input_size, num_params):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, num_params),
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Initialize controller
            >>> chorus = MultiVoiceChorus(sample_rate=44100, num_voices=3)
            >>> num_params = chorus.count_num_parameters()  # 7 parameters (4 base + 3 gains)
            >>> controller = MultiChorusController(input_size=16, num_params=num_params)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = controller(features)
            >>> output = chorus(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register default parameters for the multi-voice chorus.
    
        Sets up:
            delay_ms: Base delay time (1.0 to 10.0 ms)
            rate: LFO modulation rate (0.1 to 10.0 Hz)
            depth: Modulation intensity (0.0 to 0.25)
            mix: Wet/dry balance (0.0 to 1.0)
            gi: Gain for each voice i (0.0 to 1.0)
        """
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

    Args:
    sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
    num_voices (int): Number of chorus voices. Defaults to 2.

    Attributes:
    num_voices (int): Number of active chorus voices
    sample_rate (int): Audio sample rate in Hz

    Parameters Details:
    delay_ms: Base delay time 
        - Range: 1.0 to 10.0 ms
        - Controls center delay time
        - Shorter delays for tighter effect
        
    rate: LFO modulation frequency
        - Range: 0.1 to 10.0 Hz
        - Controls modulation speed
        - Lower values for gentle chorusing
        
    depth: Modulation intensity
        - Range: 0.0 to 0.25
        - Controls amount of detuning
        - Affects richness of chorus
        
    mix: Wet/dry balance
        - Range: 0.0 to 1.0
        - 0.0: Only clean signal
        - 1.0: Only processed signal
        
    For each voice i:
        gi: Voice gain
            - Range: 0.0 to 1.0
            - Controls level of each voice
            
        pani: Voice stereo position
            - Range: -1.0 to 1.0
            - -1.0: Full left
            - 0.0: Center
            - 1.0: Full right

    Warning:
    When using with neural networks:
        - norm_params must be in range [0, 1]
        - Parameters will be automatically mapped to ranges
        - Ensure network output is properly normalized (e.g., using sigmoid)
        - Parameter order must match _register_default_parameters()
        - Total parameters = 4 + 2*num_voices (base params + gain/pan per voice)

    Examples:
    Basic DSP Usage:
        >>> # Create a stereo chorus with 2 voices
        >>> chorus = StereoChorus(
        ...     sample_rate=44100,
        ...     num_voices=2
        ... )
        >>> # Process with musical settings
        >>> output = chorus(input_audio, dsp_params={
        ...     'delay_ms': 5.0,    # Base delay
        ...     'rate': 1.5,        # Modulation rate
        ...     'depth': 0.15,      # Moderate detuning
        ...     'mix': 0.7,         # Mostly wet
        ...     'g0': 1.0,          # Full voice 1
        ...     'pan0': -0.7,       # Voice 1 left
        ...     'g1': 0.8,          # Reduced voice 2
        ...     'pan1': 0.7         # Voice 2 right
        ... })

    Neural Network Control:
        >>> # Simple parameter prediction
        >>> class StereoChorusController(nn.Module):
        ...     def __init__(self, input_size, num_params):
        ...         super().__init__()
        ...         self.net = nn.Sequential(
        ...             nn.Linear(input_size, 32),
        ...             nn.ReLU(),
        ...             nn.Linear(32, num_params),
        ...             nn.Sigmoid()  # Ensures output is in [0,1] range
        ...         )
        ...     
        ...     def forward(self, x):
        ...         return self.net(x)
        >>> 
        >>> # Initialize controller
        >>> chorus = StereoChorus(sample_rate=44100, num_voices=2)
        >>> num_params = chorus.count_num_parameters()  # 8 parameters (4 base + 2*2 voice params)
        >>> controller = StereoChorusController(input_size=16, num_params=num_params)
        >>> 
        >>> # Process with features
        >>> features = torch.randn(batch_size, 16)  # Audio features
        >>> norm_params = controller(features)
        >>> output = chorus(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register default parameters for the stereo chorus.
    
        Sets up:
            delay_ms: Base delay time (1.0 to 10.0 ms)
            rate: LFO modulation rate (0.1 to 10.0 Hz)
            depth: Modulation intensity (0.0 to 0.25)
            mix: Wet/dry balance (0.0 to 1.0)
            
            For each voice i:
                gi: Voice gain (0.0 to 1.0)
                pani: Voice stereo position (-1.0 to 1.0)
        """
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
        """Initialize the stereo chorus processor.
    
        Args:
            sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
            num_voices (int): Number of chorus voices. Defaults to 2.
        """
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
    
    



