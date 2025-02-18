import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.phase import unwrap_phase



# (time domain shift = freq domain phase shift)
# padding, unwrap 
# padding for solving aliasing 
# unwraping for solving phase discontinuity 
# ref: https://ccrma.stanford.edu/~jos/fp3/Phase_Unwrapping.html

# Basic Delay 
class BasicDelay(ProcessorsBase):
    """Differentiable implementation of a single-tap delay line.
    
    This processor implements a basic digital delay line using frequency-domain processing
    for precise, artifact-free time delays. It creates a single echo of the input signal
    with controllable delay time and mix level.

    The delay is implemented in the frequency domain using the time-shift property:

    .. math::

        Y(\\omega) = X(\\omega)e^{-j\\omega\\tau}

    where:
        - X(ω) is the input spectrum
        - Y(ω) is the delayed spectrum
        - τ is the delay time in seconds
        - Phase is unwrapped to ensure continuous delay response

    Args:
        sample_rate (int): Audio sample rate in Hz

    Parameters Details:
        delay_ms: Echo delay time
            - Range: 0.1 to 3000.0 milliseconds
            - Controls time offset between original and delayed signal
            - Minimum value ensures stable processing
            - Maximum value set for practical buffer sizes
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only original signal
            - 1.0: Only delayed signal
            - Linear crossfade between original and delayed signals

    Note:
        - Uses FFT-based delay for precise time shifting
        - Phase unwrapping prevents discontinuities in delay
        - Automatic padding handles all delay times
        - More efficient than time-domain implementation for longer delays

    Examples:
        Basic DSP Usage:
            >>> # Create a basic delay
            >>> delay = BasicDelay(sample_rate=44100)
            >>> # Process audio
            >>> output = delay(input_audio, dsp_params={
            ...     'delay_ms': 500.0,  # Half-second delay
            ...     'mix': 0.5          # Equal mix of dry and wet
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class DelayController(nn.Module):
            ...     def __init__(self, input_size):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, 2),  # 2 parameters: delay and mix
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Process with features
            >>> controller = DelayController(input_size=16)
            >>> features = torch.randn(batch_size, 16)
            >>> norm_params = controller(features)
            >>> output = delay(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register delay time and mix parameters.
        
        Sets up two parameters:
            - delay_ms: Delay time in milliseconds (0.1 to 3000.0)
            - mix: Wet/dry mix ratio (0.0 to 1.0)
        """
        self.params = {
            'delay_ms': EffectParam(min_val=0.1, max_val=3000.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the delay line.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, torch.Tensor], optional): Direct DSP parameters.
                If provided, norm_params must be None.
                
        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
            
        Processing steps:
            1. Parameter validation and mapping
            2. Zero-pad input for delay buffer
            3. Convert to frequency domain
            4. Calculate and apply phase shift
            5. Convert back to time domain
            6. Mix delayed signal with original
            
        Note:
            Delay is implemented in frequency domain for efficiency and precision.
            Phase unwrapping ensures continuous delay response.
        """
        # Set proper configuration
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        # get parameters 
        delay_ms, mix = params['delay_ms'], params['mix']
        
        # Padding 
        max_delay_samples = int(torch.max(delay_ms) * self.sample_rate / 1000)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, 0))
        
        # Convert to frequency domain
        X = torch.fft.rfft(x_padded)
        
        # Phase calculation with unwrapping
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate).to(x.device)
        phase = -2 * np.pi * freqs * delay_ms.view(-1, 1, 1) / 1000
        phase = unwrap_phase(phase, dim=-1)
        
        # Apply phase shift
        X_delayed = X * torch.exp(1j * phase).to(X.dtype)
        
        # IFFT and trim padding
        x_delayed = torch.fft.irfft(X_delayed, n=x_padded.shape[-1])[:, :, max_delay_samples:]
        
        mix = mix.unsqueeze(-1).unsqueeze(-1)
        return (1 - mix) * x + mix * x_delayed


# Add feedback 
class BasicFeedbackDelay(ProcessorsBase):
    """Differentiable implementation of a feedback delay line.
    
    This processor implements a delay line with feedback and feedforward paths, creating
    multiple decaying echoes. The implementation uses frequency-domain processing and 
    a feedback-feedforward structure for flexible echo patterns.

    The transfer function of the system is:

    .. math::

        H(z) = \\frac{z^{-N} + g_{ff} - g_{fb}}{z^{-N} - g_{fb}}

    where:
        - z^(-N) represents the delay of N samples
        - g_ff is the feedforward gain
        - g_fb is the feedback gain
        - System stability is ensured by limiting |g_fb| < 1

    Processing Chain:
        1. Zero-pad input for delay buffer
        2. Convert to frequency domain
        3. Calculate phase shift (z^-N term)
        4. Apply transfer function H(z)
        5. Convert back to time domain
        6. Mix processed signal with original

    Args:
        sample_rate (int): Audio sample rate in Hz

    Parameters Details:
        delay_ms: Echo delay time
            - Range: 0.1 to 3000.0 milliseconds
            - Controls time between successive echoes
            - Determines rhythmic pattern of echoes
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only original signal
            - 1.0: Only processed signal
            
        fb_gain: Feedback gain
            - Range: 0.0 to 0.99
            - Controls decay rate of echoes
            - Higher values create longer decay times
            - Clamped to ±0.99 for stability
            
        ff_gain: Feedforward gain
            - Range: 0.0 to 0.99
            - Controls level of direct delayed signal
            - Shapes initial echo response
            - Independent of feedback path

    Note:
        - Feedback gain is clamped to ensure system stability
        - Uses frequency domain for efficient processing
        - Epsilon added to denominator to prevent division by zero
        - Multiple echoes decay exponentially with feedback
        - Combined feedforward/feedback structure allows flexible echo patterns

    Warning:
        - High feedback gains (>0.9) can create very long decay times
        - Monitor output levels when using high feedback values
        - Feedback near ±1.0 can cause self-oscillation

    Examples:
        Basic DSP Usage:
            >>> # Create a feedback delay
            >>> delay = BasicFeedbackDelay(sample_rate=44100)
            >>> # Process with rhythmic echoes
            >>> output = delay(input_audio, dsp_params={
            ...     'delay_ms': 250.0,  # Quarter note at 120 BPM
            ...     'mix': 0.5,         # Equal mix
            ...     'fb_gain': 0.7,     # Moderate feedback
            ...     'ff_gain': 0.8      # Strong initial echo
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class FeedbackDelayController(nn.Module):
            ...     def __init__(self, input_size):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, 4),  # 4 parameters
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Process with features
            >>> controller = FeedbackDelayController(input_size=16)
            >>> features = torch.randn(batch_size, 16)
            >>> norm_params = controller(features)
            >>> output = delay(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register delay, mix, and gain parameters.
        
        Sets up four parameters:
            - delay_ms: Delay time in milliseconds (0.1 to 3000.0)
            - mix: Wet/dry mix ratio (0.0 to 1.0)
            - fb_gain: Feedback gain (0.0 to 0.99)
            - ff_gain: Feedforward gain (0.0 to 0.99)
        """
        self.params = {
            'delay_ms': EffectParam(min_val=0.1, max_val=3000.0),
            'mix': EffectParam(min_val=0, max_val=1.0),
            'fb_gain': EffectParam(min_val=0.0, max_val=0.99),
            'ff_gain': EffectParam(min_val=0.0, max_val=0.99)
        }
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the feedback delay line.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, torch.Tensor], optional): Direct DSP parameters.
                If provided, norm_params must be None.
                
        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
            
        Processing steps:
            1. Parameter validation and mapping
            2. Zero-pad input for delay buffer
            3. Convert to frequency domain
            4. Calculate transfer function H(z)
            5. Apply transfer function
            6. Convert back to time domain
            7. Mix processed signal with original
            
        Note:
            Implementation uses frequency domain transfer function
            for efficient computation of feedback structure.
            Feedback gain is clamped to ensure stability.
        """
        # Set proper configuration
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        delay_ms = params['delay_ms'].view(-1, 1, 1)
        g_fb = params['fb_gain'].view(-1, 1, 1)
        g_fb = torch.clamp(g_fb, -0.99, 0.99)
        g_ff = params['ff_gain'].view(-1, 1, 1)
        mix = params['mix'].view(-1, 1, 1)
        
        # padding 
        max_delay_samples = int(torch.max(delay_ms) * self.sample_rate / 1000)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, 0))
       
        # freq domain 
        X = torch.fft.rfft(x_padded)
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate).to(x.device)
        phase = -2 * np.pi * freqs * delay_ms / 1000
        phase = unwrap_phase(phase, dim=-1)
        z_n = torch.exp(1j * phase).to(X.dtype)
        
        # H(z) = (z^N + g_FF - g_FB)/(z^N - g_FB)
        eps = 1e-6
        H = (z_n + g_ff - g_fb) / (z_n - g_fb + eps)
        X_delayed = X * H
        
        x_delayed = torch.fft.irfft(X_delayed, n=x_padded.shape[-1])[:, :, max_delay_samples:]

        return (1 - mix) * x + mix * x_delayed

# Identical to the basic delay but the delay_ms is much shorter 
class SlapbackDelay(BasicDelay):
    """Differentiable implementation of a slapback delay effect.
    
    This processor extends BasicDelay to create a specialized short delay effect
    that emulates the distinctive "doubling" sound popularized in 1950s recordings.
    The delay time range is specifically restricted to create the characteristic
    slapback effect.

    The processor uses the same frequency-domain implementation as BasicDelay:

    .. math::

        Y(\\omega) = X(\\omega)e^{-j\\omega\\tau}

    where τ is restricted to 40-120ms for the slapback effect.

    Delay Time Ranges:
        - 40-80ms: Tight doubling effect
        - 80-120ms: Subtle ambience
        These ranges are chosen based on psychoacoustic research
        and historical usage in classic recordings.

    Args:
        sample_rate (int): Audio sample rate in Hz

    Parameters Details:
        delay_ms: Slapback delay time
            - Range: 40.0 to 120.0 milliseconds
            - Shorter range than BasicDelay for specific effect
            - 40-80ms: Creates tight doubling
            - 80-120ms: Adds natural space
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only original signal
            - 1.0: Only delayed signal
            - Typical settings: 0.3-0.5 for classic sound

    Note:
        - Inherits all processing methods from BasicDelay
        - Only modifies parameter ranges for specialized use
        - Particularly effective on:
            - Vocals (creates natural doubling)
            - Electric guitar (adds depth)
            - Snare drums (enhances attack)
        - No feedback to maintain clarity of effect

    Examples:
        Basic DSP Usage:
            >>> # Create a slapback delay
            >>> delay = SlapbackDelay(sample_rate=44100)
            >>> # Process with classic settings
            >>> output = delay(input_audio, dsp_params={
            ...     'delay_ms': 60.0,  # Tight doubling effect
            ...     'mix': 0.4         # Subtle enhancement
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class SlapbackController(nn.Module):
            ...     def __init__(self, input_size):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, 2),  # 2 parameters: delay and mix
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Process with features
            >>> controller = SlapbackController(input_size=16)
            >>> features = torch.randn(batch_size, 16)
            >>> norm_params = controller(features)
            >>> output = delay(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register parameters with slapback-specific ranges.
        
        Modifies the delay time range from BasicDelay to:
            - delay_ms: 40.0 to 120.0 ms (slapback range)
            - mix: 0.0 to 1.0 (unchanged from BasicDelay)
            
        Note:
            These ranges are specifically chosen for the
            characteristic slapback doubling effect.
        """
        self.params = {
            'delay_ms': EffectParam(min_val=40.0, max_val=120.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
    

