import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.phase import unwrap_phase


class PingPongDelay(ProcessorsBase):
    """Differentiable implementation of a stereo ping-pong delay effect.
    
    This processor implements a stereo delay effect where echoes alternate between
    left and right channels, creating a "ping-pong" spatial pattern. The implementation
    uses a cross-coupled feedback structure in the frequency domain for precise timing
    and smooth transitions.

    Implementation is based on: 
    
    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [2] Smith, Julius O. "Digital Audio Effects." 
            https://ccrma.stanford.edu/~jos/fp3/Phase_Unwrapping.html
    
    The system is described by coupled transfer functions:

    .. math::

        H_{11}(z) = \\frac{1}{1 - b_1b_2z^{-2N}}

        H_{12}(z) = \\frac{b_1z^{-N}}{1 - b_1b_2z^{-2N}}

        H_{21}(z) = \\frac{b_2z^{-N}}{1 - b_1b_2z^{-2N}}

        H_{22}(z) = \\frac{b_1b_2z^{-2N}}{1 - b_1b_2z^{-2N}}

    where:
        - z^(-N) represents the base delay
        - b1, b2 are feedback gains for each channel
        - System stability ensured by |b1*b2| < 1

    Processing Chain:
        1. Zero-pad stereo input for delay buffer
        2. Convert to frequency domain
        3. Calculate cross-coupled transfer functions
        4. Apply transfers to each channel
        5. Convert back to time domain
        6. Mix processed signal with original

    Args:
        sample_rate (int): Audio sample rate in Hz
        param_range (Dict[str, EffectParam], optional): Parameter ranges.

    Parameters Details:
        delay_ms: Base delay time
            - Range: 0.1 to 3000.0 milliseconds
            - Controls time between alternating echoes
            - Each bounce takes this amount of time
            
        feedback_ch1: Left channel feedback gain
            - Range: 0.0 to 0.99
            - Controls decay of left-to-right echoes
            - Higher values create longer decay times
            
        feedback_ch2: Right channel feedback gain
            - Range: 0.0 to 0.99
            - Controls decay of right-to-left echoes
            - Can differ from ch1 for asymmetric patterns
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only original signal
            - 1.0: Only processed signal

    Note:
        - Uses FFT-based delay for precise time shifting
        - Phase unwrapping prevents discontinuities
        - Automatic padding handles all delay times
        - Particularly effective for:
            - Creating rhythmic spatial patterns
            - Adding stereo width and movement
            - Building complex stereo textures
        - System stability is maintained by gain limits

    Examples:
        Basic DSP Usage:
            >>> # Create a ping-pong delay
            >>> delay = PingPongDelay(sample_rate=44100)
            >>> # Process with rhythmic spatial echoes
            >>> output = delay(input_audio, dsp_params={
            ...     'delay_ms': 250.0,     # Quarter note at 120 BPM
            ...     'feedback_ch1': 0.7,   # Left to right decay
            ...     'feedback_ch2': 0.7,   # Right to left decay
            ...     'mix': 0.5            # Equal mix of dry and wet
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class PingPongController(nn.Module):
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
            >>> controller = PingPongController(input_size=16)
            >>> features = torch.randn(batch_size, 16)
            >>> norm_params = controller(features)
            >>> output = delay(input_audio, norm_params=norm_params)
    """
    def _register_default_parameters(self):
        """Register delay time, feedback, and mix parameters.
        
        Sets up four parameters:
            - delay_ms: Base delay time (0.1 to 3000.0 ms)
            - feedback_ch1: Left channel feedback (0.0 to 0.99)
            - feedback_ch2: Right channel feedback (0.0 to 0.99)
            - mix: Wet/dry mix ratio (0.0 to 1.0)
        """
        self.params = {
            'delay_ms': EffectParam(min_val=0.1, max_val=3000.0),
            'feedback_ch1': EffectParam(min_val=0.0, max_val=0.99),
            'feedback_ch2': EffectParam(min_val=0.0, max_val=0.99),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None , dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the ping-pong delay.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 2, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'delay_ms': Base delay time in milliseconds (0 to 1)
                - 'feedback_ch1': Left channel feedback (0 to 1)
                - 'feedback_ch2': Right channel feedback (0 to 1)
                - 'mix': Wet/dry balance (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify ping-pong parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor of same shape as input. Shape: (batch, 2, samples)
            
        Raises:
            AssertionError: If input is not stereo (2 channels)
        """
        # Set proper configuration
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        delay_ms = params['delay_ms'].view(-1, 1, 1)
        b1 = params['feedback_ch1'].view(-1, 1, 1)
        b2 = params['feedback_ch2'].view(-1, 1, 1)
        mix = params['mix'].view(-1, 1, 1)
        
        b, ch, s = x.shape
        assert ch == 2, "Input must be stereo"
        
        max_delay_samples = max(
            1,
            int(torch.max(delay_ms) * self.sample_rate / 1000)
        )
        # Calculate FFT size (next power of 2 for efficiency)
        fft_size = 2 ** int(np.ceil(np.log2(x.shape[-1] + max_delay_samples)))
        # Pad input signal to FFT size
        pad_right = fft_size - (x.shape[-1] + max_delay_samples)
        x_padded = torch.nn.functional.pad(x, (max_delay_samples, pad_right))
        
        X = torch.fft.rfft(x_padded, n=fft_size)
        freqs = torch.fft.rfftfreq(x_padded.shape[-1], 1/self.sample_rate).to(x.device)
        phase = -2 * np.pi * freqs * delay_ms / 1000
        phase = unwrap_phase(phase, dim=-1)
        z_n = torch.exp(1j * phase).to(X.dtype)
        
        eps = 1e-6
        den = 1 - b1 * b2 * z_n * z_n + eps
        
        # Modified transfer functions for ping-pong behavior
        H11 = 1 /den  # Direct path for left
        H12 = b1 * z_n / den  # Left to right (single delay)
        H21 = b2 * z_n / den  # Right to left (single delay)
        H22 = b1 * b2 * z_n * z_n / den  # Right to right (double delay through feedback)
        
        Y1 = H11 * X[:, 0:1] + H12 * X[:, 1:2]
        Y2 = H21 * X[:, 0:1] + H22 * X[:, 1:2]
        
        Y = torch.cat([Y1, Y2], dim=1)
        # y = torch.fft.irfft(Y, n=x_padded.shape[-1])[:, :, max_delay_samples:]
        y = torch.fft.irfft(Y, n=fft_size)
        y = y[..., max_delay_samples:max_delay_samples + x.shape[-1]]
        
        return (1 - mix) * x + mix * y
    