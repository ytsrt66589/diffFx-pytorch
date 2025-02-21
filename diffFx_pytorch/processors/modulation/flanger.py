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

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.

    Attributes:
        sample_rate (int): Audio sample rate in Hz

    Parameters Details:
        delay_ms: Base delay time 
            - Range: 1.0 to 10.0 ms
            - Controls center delay time
            - Very short delays for flanger effect
            
        rate: LFO modulation frequency
            - Range: 0.1 to 2.0 Hz
            - Controls modulation speed
            - Lower values create slow sweeps
            - Higher values for faster effects
            
        depth: Modulation intensity
            - Range: 0.0 to 1.0
            - Controls sweep width
            - Affects intensity of effect
            
        mix: Wet/dry balance
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only flanged signal

    Note:
        The processor supports the following features:
            - Variable delay implementation
            - Smooth LFO modulation
            - Phase-coherent processing
            - Automatic buffer size handling
            - Efficient batch processing

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to ranges
            - Ensure network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()

    Examples:
        Basic DSP Usage:
            >>> # Create a flanger effect
            >>> flanger = Flanger(
            ...     sample_rate=44100
            ... )
            >>> # Process with musical settings
            >>> output = flanger(input_audio, dsp_params={
            ...     'delay_ms': 5.0,    # 5ms base delay
            ...     'rate': 0.5,        # 0.5 Hz modulation
            ...     'depth': 0.7,       # Strong sweep
            ...     'mix': 0.6          # 60% wet
            ... })

        Neural Network Control:
            >>> # Simple parameter prediction
            >>> class FlangerController(nn.Module):
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
            >>> flanger = Flanger(sample_rate=44100)
            >>> num_params = flanger.count_num_parameters()  # 4 parameters
            >>> controller = FlangerController(input_size=16, num_params=num_params)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = controller(features)
            >>> output = flanger(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register default parameters for the flanger effect.
    
        Sets up:
            delay_ms: Base delay time (1.0 to 10.0 ms)
            rate: LFO modulation rate (0.1 to 2.0 Hz)
            depth: Modulation intensity (0.0 to 1.0)
            mix: Wet/dry balance (0.0 to 1.0)
        """
        self.params = {
            'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
            'rate': EffectParam(min_val=0.1, max_val=2.0),         # More musical range
            'depth': EffectParam(min_val=0.0, max_val=1.0),        # Full range
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self._register_default_parameters()
        
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
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
        # Get parameters
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
        max_delay_samples = int(torch.max(delay_ms) * self.sample_rate / 1000.0)
        delay_center = delay_ms / 1000.0 * self.sample_rate # samples 
        
        # Generate time base for LFO
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        
        # Generate LFO with batch dimension
        # phase = 2 * math.pi * rate * time.view(1, 1, -1)  # (1, 1, n_samples)
        delay_lfo = torch.sin(2 * math.pi * rate * time.view(1, 1, -1))  # (batch, 1, n_samples)
        
        # Calculate delay values
        delay_value = delay_lfo * (depth * delay_center) + delay_center # 
        # print('> delay_value: ', delay_value)
        delay_phase = delay_value / max_delay_samples
        
        # Expand phase for all channels
        delay_phase = delay_phase.expand(-1, n_ch, -1)  # (batch, channel, n_samples)
        
        # Apply variable delay
        delayed = variable_delay(delay_phase, x, buf_size=math.ceil(max_delay_samples))
        
        # Mix dry and wet signals
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

    Args:
    sample_rate (int): Audio sample rate in Hz. Defaults to 44100.

    Attributes:
    sample_rate (int): Audio sample rate in Hz

    Parameters Details:
    delay_ms: Base delay time 
        - Range: 1.0 to 10.0 ms
        - Controls center delay time
        - Very short delays for flanger effect
        
    rate: LFO modulation frequency
        - Range: 0.1 to 2.0 Hz
        - Controls modulation speed
        - Lower values create slow stereo sweeps
        
    depth: Modulation intensity
        - Range: 0.0 to 1.0
        - Controls stereo sweep width
        - Affects intensity of effect
        
    mix: Wet/dry balance
        - Range: 0.0 to 1.0
        - 0.0: Only clean signal
        - 1.0: Only flanged signal

    Note:
    The processor supports the following features:
        - Quadrature LFOs for true stereo
        - Independent channel processing
        - Phase-coherent stereo field
        - Automatic buffer size handling
        - Efficient batch processing

    Warning:
    When using with neural networks:
        - norm_params must be in range [0, 1]
        - Parameters will be automatically mapped to ranges
        - Ensure network output is properly normalized (e.g., using sigmoid)
        - Parameter order must match _register_default_parameters()
        - Input must be stereo (2 channels)

    Examples:
    Basic DSP Usage:
        >>> # Create a stereo flanger
        >>> flanger = StereoFlanger(
        ...     sample_rate=44100
        ... )
        >>> # Process with musical settings
        >>> output = flanger(input_audio, dsp_params={
        ...     'delay_ms': 5.0,    # 5ms base delay
        ...     'rate': 0.5,        # 0.5 Hz modulation
        ...     'depth': 0.7,       # Strong sweep
        ...     'mix': 0.6          # 60% wet
        ... })

    Neural Network Control:
        >>> # Simple parameter prediction
        >>> class StereoFlangerController(nn.Module):
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
        >>> flanger = StereoFlanger(sample_rate=44100)
        >>> num_params = flanger.count_num_parameters()  # 4 parameters
        >>> controller = StereoFlangerController(input_size=16, num_params=num_params)
        >>> 
        >>> # Process with features
        >>> features = torch.randn(batch_size, 16)  # Audio features
        >>> norm_params = controller(features)
        >>> output = flanger(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register default parameters for the stereo flanger effect.
   
        Sets up:
            delay_ms: Base delay time (1.0 to 10.0 ms)
            rate: LFO modulation rate (0.1 to 2.0 Hz)
            depth: Modulation intensity (0.0 to 1.0)
            mix: Wet/dry balance (0.0 to 1.0)
        """
        self.params = {
            'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
            'rate': EffectParam(min_val=0.1, max_val=2.0),         # More musical range
            'depth': EffectParam(min_val=0.0, max_val=1.0),        # Full range
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self._register_default_parameters()
        
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
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
        
        # Map parameters with correct shapes
        delay_ms = params['delay_ms'].view(-1, 1, 1)    # (batch, 1, 1)
        rate = params['rate'].view(-1, 1, 1)            # (batch, 1, 1)
        depth = params['depth'].view(-1, 1, 1)          # (batch, 1, 1)
        mix = params['mix'].view(-1, 1, 1)              # (batch, 1, 1)
        
        # Calculate maximum delay in samples
        max_delay_samples = int(torch.max(delay_ms) * self.sample_rate / 1000.0)
        delay_center = delay_ms / 1000.0 * self.sample_rate # samples 
        
        # Generate time base for LFO
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        
        # Generate quadrature LFOs (90 degrees phase difference)
        phase_left = 2 * math.pi * rate * time.view(1, 1, -1)
        phase_right = phase_left + math.pi/2  # 
        
        # Generate left and right channel LFOs
        delay_lfo_left = torch.sin(phase_left)   # LFO
        delay_lfo_right = torch.sin(phase_right) # LFO
        
        # Stack LFOs for stereo processing
        delay_lfo = torch.cat([delay_lfo_left, delay_lfo_right], dim=1)  # (batch, 2, samples)
        
        # Calculate delay values (now for both channels)
        delay_value = delay_lfo * (depth * delay_center) + delay_center
        delay_phase = delay_value / max_delay_samples
        
        # Apply stereo delay
        delayed = variable_delay(delay_phase, x, buf_size=math.ceil(max_delay_samples))
        
        # Mix dry and wet signals (same as before)
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

    Args:
    sample_rate (int): Audio sample rate in Hz. Defaults to 44100.

    Attributes:
    sample_rate (int): Audio sample rate in Hz

    Parameters Details:
    delay_ms: Base delay time 
        - Range: 1.0 to 10.0 ms
        - Controls center delay time
        - Very short delays for flanger effect
        
    rate: LFO modulation frequency
        - Range: 0.1 to 10.0 Hz
        - Controls modulation speed
        - Lower values create slow sweeps
        
    depth: Modulation intensity
        - Range: 0.0 to 0.25
        - Controls sweep width
        - Affects intensity of effect
        
    feedback: Feedback amount
        - Range: 0.0 to 0.7
        - Controls resonance intensity
        - Higher values create metallic sound
        
    mix: Wet/dry balance
        - Range: 0.0 to 1.0
        - 0.0: Only clean signal
        - 1.0: Only flanged signal

    Note:
    The processor supports the following features:
        - Variable delay implementation
        - Feedback path processing
        - Resonant frequency peaks
        - Automatic stability control
        - Efficient batch processing

    Warning:
    When using with neural networks:
        - norm_params must be in range [0, 1]
        - Parameters will be automatically mapped to ranges
        - Ensure network output is properly normalized (e.g., using sigmoid)
        - Parameter order must match _register_default_parameters()
        - High feedback can create intense resonance

    Examples:
    Basic DSP Usage:
        >>> # Create a feedback flanger
        >>> flanger = FeedbackFlanger(
        ...     sample_rate=44100
        ... )
        >>> # Process with musical settings
        >>> output = flanger(input_audio, dsp_params={
        ...     'delay_ms': 5.0,     # 5ms base delay
        ...     'rate': 0.5,         # 0.5 Hz modulation
        ...     'depth': 0.15,       # Moderate sweep
        ...     'feedback': 0.4,     # Medium resonance
        ...     'mix': 0.6           # 60% wet
        ... })

    Neural Network Control:
        >>> # Simple parameter prediction
        >>> class FlangerController(nn.Module):
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
        >>> flanger = FeedbackFlanger(sample_rate=44100)
        >>> num_params = flanger.count_num_parameters()  # 5 parameters
        >>> controller = FlangerController(input_size=16, num_params=num_params)
        >>> 
        >>> # Process with features
        >>> features = torch.randn(batch_size, 16)  # Audio features
        >>> norm_params = controller(features)
        >>> output = flanger(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register default parameters for the feedback flanger effect.

        Sets up:
            delay_ms: Base delay time (1.0 to 10.0 ms)
            rate: LFO modulation rate (0.1 to 10.0 Hz)
            depth: Modulation intensity (0.0 to 0.25)
            feedback: Feedback amount (0.0 to 0.7)
            mix: Wet/dry balance (0.0 to 1.0)
        """
        self.params = {
            'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
            'rate': EffectParam(min_val=0.1, max_val=10.0),         # More musical range
            'depth': EffectParam(min_val=0.0, max_val=0.25),        # Full range
            'feedback': EffectParam(min_val=0.0, max_val=0.7),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self._register_default_parameters()
        
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
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
        
        # Map parameters with correct shapes
        delay_ms = params['delay_ms'].view(-1, 1, 1)    # (batch, 1, 1)
        rate = params['rate'].view(-1, 1, 1)            # (batch, 1, 1)
        depth = params['depth'].view(-1, 1, 1)          # (batch, 1, 1)
        mix = params['mix'].view(-1, 1, 1)              # (batch, 1, 1)
        feedback = params['feedback'].view(-1, 1, 1)
        
        # Calculate maximum delay in samples
        max_delay_samples = int(torch.max(delay_ms) * self.sample_rate / 1000.0)
        delay_center = delay_ms / 1000.0 * self.sample_rate # samples 
        
        # Generate time base for LFO
        time = torch.linspace(0, n_samples/self.sample_rate, n_samples, device=device)
        
        # Generate LFO with batch dimension
        phase = 2 * math.pi * rate * time.view(1, 1, -1)  # (1, 1, n_samples)
        delay_lfo = torch.sin(phase)  # (batch, 1, n_samples)
        
        # Calculate delay values
        delay_value = delay_lfo * (depth * delay_center) + delay_center # 
        delay_phase = delay_value / max_delay_samples
        
        # Expand phase for all channels
        delay_phase = delay_phase.expand(-1, n_ch, -1)  # (batch, channel, n_samples)
        
        # 初始化 feedback buffer
        delayed = x
        # Apply variable delay
        delayed = variable_delay(delay_phase, x + feedback * delayed, buf_size=math.ceil(max_delay_samples))
        
        # Mix dry and wet signals
        return mix * delayed + (1 - mix) * x
    
    
