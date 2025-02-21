import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..filters import DCFilter


class BaseDistortion(ProcessorsBase):
    """Base class for implementing various distortion effects with optional waveshaping controls.

    This class provides a foundation for implementing different types of distortion effects,
    with optional waveshaping parameters for more detailed control over the distortion characteristics.
    It includes pre/post gain staging, DC bias control, and automatic DC filtering when in shaping mode.

    Args:
        sample_rate (int): Audio sample rate in Hz
        shaping_mode (bool): Whether to enable additional waveshaping controls. Defaults to False.
            When True, enables:
            - Pre-gain control
            - DC bias adjustment
            - Post-gain control
            - Automatic DC filtering

    Default Parameters:
        Basic Mode (shaping_mode=False):
            mix: Wet/dry mix ratio
                - Range: 0.0 to 1.0
                - 0.0: Only clean signal
                - 1.0: Only distorted signal

        Shaping Mode (shaping_mode=True):
            Additional parameters:
            pre_gain_db: Input gain before distortion
                - Range: -24.0 to 24.0 dB
                - Controls amount of drive into distortion
                - Higher values create more saturation
                
            post_gain_db: Output gain after distortion
                - Range: -24.0 to 0.0 dB
                - Compensates for level changes
                - Prevents output clipping
                
            dc_bias: DC offset added before distortion
                - Range: -0.2 to 0.2
                - Controls asymmetric clipping
                - Affects harmonic content

    Note:
        - Subclasses must implement _apply_distortion method
        - DC filtering is automatically applied in shaping mode
        - Parameters can be controlled via norm_params or dsp_params
        - Additional parameters can be added through _add_specific_parameters

    Example:
        ```python
        class CustomDistortion(BaseDistortion):
            def _add_specific_parameters(self):
                self.params['drive'] = EffectParam(min_val=1.0, max_val=10.0)
                
            def _apply_distortion(self, x, params):
                drive = params['drive'].unsqueeze(-1).unsqueeze(-1)
                return torch.tanh(drive * x)
        ```
    """
    def __init__(self, sample_rate, shaping_mode=False):
        """Initialize the distortion processor.
    
        Args:
            sample_rate (int): Audio sample rate in Hz
            shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        """
        self.shaping_mode = shaping_mode
        super().__init__(sample_rate)
        
        if self.shaping_mode:
            self.dc_filter = DCFilter(sample_rate, learnable=False)
    
    def _register_default_parameters(self):
        """Register default parameters for the distortion processor.
    
        Sets up:
            Basic Mode:
                - mix: Wet/dry balance (0.0 to 1.0)
                
            Shaping Mode:
                - pre_gain_db: Input gain (-24.0 to 24.0 dB)
                - post_gain_db: Output gain (-24.0 to 0.0 dB)
                - dc_bias: DC offset (-0.2 to 0.2)
        """
        base_params = {
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
        shaping_params = {
            'pre_gain_db': EffectParam(min_val=-24.0, max_val=24.0),
            'post_gain_db': EffectParam(min_val=-24.0, max_val=0.0),
            'dc_bias': EffectParam(min_val=-0.2, max_val=0.2),
        }
        
        if self.shaping_mode:
            self.params = {**base_params, **shaping_params}
        else:
            self.params = base_params
            
        # Add any additional parameters specific to the distortion type
        self._add_specific_parameters()
    
    def _add_specific_parameters(self):
        """Add parameters specific to each distortion type.
    
        This method should be overridden by subclasses to add parameters
        specific to their distortion implementation.
        
        Example:
            ```python
            def _add_specific_parameters(self):
                self.params['drive'] = EffectParam(min_val=1.0, max_val=10.0)
            ```
        """
        pass
    
    def _apply_distortion(self, x):
        """Apply the distortion transfer function to the input signal.
    
        Args:
            x (torch.Tensor): Input audio tensor
            
        Returns:
            torch.Tensor: Distorted audio tensor
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the distortion effect.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Basic Mode must contain:
                    - 'mix': Wet/dry balance (0 to 1)
                Shaping Mode adds:
                    - 'pre_gain_db': Input gain (0 to 1)
                    - 'post_gain_db': Output gain (0 to 1)
                    - 'dc_bias': DC offset (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify distortion parameters as:
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
        
        # Extract parameters
        mix = params['mix']
        
        # Apply shaping if enabled
        if self.shaping_mode:
            pre_gain = 10 ** (params['pre_gain_db'] / 20.0)
            post_gain = 10 ** (params['post_gain_db'] / 20.0)
            pre_gain = pre_gain.unsqueeze(-1).unsqueeze(-1)
            post_gain = post_gain.unsqueeze(-1).unsqueeze(-1)
            dc_bias = params['dc_bias'].unsqueeze(-1).unsqueeze(-1)
            x_driven = pre_gain * x + dc_bias
        else:
            x_driven = x
        
        # Apply distortion
        x_distorted = self._apply_distortion(x_driven, params)
        
        # Apply post-processing
        if self.shaping_mode:
            x_processed = post_gain * x_distorted
            x_processed = self.dc_filter(x_processed, None, None)
        else:
            x_processed = x_distorted
        
        # Apply mix
        mix = mix.unsqueeze(-1).unsqueeze(-1)
        return (1 - mix) * x + mix * x_processed

class TanHDist(BaseDistortion):
    """Differentiable implementation of hyperbolic tangent distortion.

    This processor implements smooth distortion using the hyperbolic tangent (tanh) function 
    for waveshaping. It provides analog-style saturation with natural compression characteristics
    and gradual onset of distortion.

    The transfer function is:

    .. math::

        y = tanh(x)

    where x is the input signal (optionally pre-gained and DC biased in shaping mode).

    Args:
    sample_rate (int): Audio sample rate in Hz
    shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        When True, enables:
        - Pre-gain control
        - DC bias adjustment
        - Post-gain control
        - Automatic DC filtering

    Parameters Details:
    Basic Mode (shaping_mode=False):
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only distorted signal

    Shaping Mode (shaping_mode=True):
        Adds:
        pre_gain_db: Input gain before distortion
            - Range: -24.0 to 24.0 dB
            - Controls amount of drive into saturation
            
        post_gain_db: Output gain after distortion
            - Range: -24.0 to 0.0 dB
            - Compensates for level changes
            
        dc_bias: DC offset before distortion
            - Range: -0.2 to 0.2
            - Affects harmonic content

    Examples:
    Basic DSP Usage:
        >>> # Create a tanh distortion
        >>> dist = TanHDist(sample_rate=44100)
        >>> # Process with basic settings
        >>> output = dist(input_audio, dsp_params={
        ...     'mix': 0.7  # 70% wet signal
        ... })

    Advanced Usage (shaping_mode=True):
        >>> # Create with waveshaping controls
        >>> dist = TanHDist(sample_rate=44100, shaping_mode=True)
        >>> # Process with detailed control
        >>> output = dist(input_audio, dsp_params={
        ...     'pre_gain_db': 12.0,   # Drive into distortion
        ...     'post_gain_db': -6.0,  # Compensate output
        ...     'dc_bias': 0.1,        # Add asymmetry
        ...     'mix': 0.8             # 80% wet
        ... })
    """
    def _apply_distortion(self, x, params):
        """Apply hyperbolic tangent distortion to the input signal.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Processing parameters (not used in this implementation)
            
        Returns:
            torch.Tensor: Distorted audio tensor of same shape as input
        """
        return torch.tanh(x)

class SoftDist(BaseDistortion):
    """Differentiable implementation of soft-clipping distortion.

    This processor implements a soft-clipping distortion using a piecewise polynomial transfer function.
    It provides smooth transitions between clean and distorted signals, creating a warm overdrive
    characteristic similar to analog tube saturation.

    The transfer function is piecewise:

    .. math::

        y = \\begin{cases} 
            1.0 & x ≥ 1 \\\\
            1.5(x - x^3/3) & -1 < x < 1 \\\\
            -1.0 & x ≤ -1
        \\end{cases}

    where x is the input signal (optionally pre-gained and DC biased in shaping mode).

    Args:
    sample_rate (int): Audio sample rate in Hz
    shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        When True, enables:
        - Pre-gain control
        - DC bias adjustment
        - Post-gain control
        - Automatic DC filtering

    Parameters Details:
    Basic Mode (shaping_mode=False):
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only distorted signal

    Shaping Mode (shaping_mode=True):
        Adds:
        pre_gain_db: Input gain before distortion
            - Range: -24.0 to 24.0 dB
            - Controls amount of drive into clipping
            
        post_gain_db: Output gain after distortion
            - Range: -24.0 to 0.0 dB
            - Compensates for level changes
            
        dc_bias: DC offset before distortion
            - Range: -0.2 to 0.2
            - Affects asymmetric clipping

    Note:
    - Smooth transition at clipping boundaries
    - More musical than hard clipping
    - Generates both odd and even harmonics
    - Good for subtle to moderate overdrive

    Examples:
    Basic DSP Usage:
        >>> # Create a soft clipper
        >>> dist = SoftDist(sample_rate=44100)
        >>> # Process with basic settings
        >>> output = dist(input_audio, dsp_params={
        ...     'mix': 0.6  # 60% wet signal
        ... })

    Advanced Usage (shaping_mode=True):
        >>> # Create with waveshaping controls
        >>> dist = SoftDist(sample_rate=44100, shaping_mode=True)
        >>> # Process with detailed control
        >>> output = dist(input_audio, dsp_params={
        ...     'pre_gain_db': 18.0,   # Drive into clipping
        ...     'post_gain_db': -9.0,  # Compensate output
        ...     'dc_bias': 0.05,       # Slight asymmetry
        ...     'mix': 0.7             # 70% wet
        ... })
    """
    def _apply_distortion(self, x, params):
        """Apply soft-clipping distortion to the input signal.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Processing parameters (not used in this implementation)
            
        Returns:
            torch.Tensor: Distorted audio tensor of same shape as input
            
        Note:
            Uses a piecewise function:
            - Linear clipping for |x| ≥ 1
            - Cubic polynomial for |x| < 1
        """
        y = torch.zeros_like(x)
        
        mask_high = x >= 1
        y[mask_high] = 1.0
        
        mask_mid = (x > -1) & (x < 1)
        x_mid = x[mask_mid]
        y[mask_mid] = 1.5 * (x_mid - x_mid**3/3)
        
        mask_low = x <= -1
        y[mask_low] = -1.0
        
        return y

class HardDist(BaseDistortion):
    """Differentiable implementation of hard-clipping distortion.

    This processor implements a hard-clipping distortion that abruptly limits signals above a 
    specified threshold. It creates aggressive distortion with rich harmonic content, similar 
    to extreme transistor or diode clipping circuits.

    The transfer function is:

    .. math::

        y = \\begin{cases} 
            threshold & x > threshold \\\\
            x & -threshold ≤ x ≤ threshold \\\\
            -threshold & x < -threshold
        \\end{cases}

    where x is the input signal (optionally pre-gained and DC biased in shaping mode).

    Args:
    sample_rate (int): Audio sample rate in Hz
    shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        When True, enables:
        - Pre-gain control
        - DC bias adjustment
        - Post-gain control
        - Automatic DC filtering

    Parameters Details:
    Basic Mode (shaping_mode=False):
        threshold: Clipping threshold level
            - Range: 0.1 to 1.0
            - Lower values create more aggressive clipping
            - Higher values preserve more dynamics
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only distorted signal

    Shaping Mode (shaping_mode=True):
        Adds:
        pre_gain_db: Input gain before distortion
            - Range: -24.0 to 24.0 dB
            - Controls amount of drive into clipping
            
        post_gain_db: Output gain after distortion
            - Range: -24.0 to 0.0 dB
            - Compensates for level changes
            
        dc_bias: DC offset before distortion
            - Range: -0.2 to 0.2
            - Affects asymmetric clipping

    Examples:
    Basic DSP Usage:
        >>> # Create a hard clipper
        >>> dist = HardDist(sample_rate=44100)
        >>> # Process with basic settings
        >>> output = dist(input_audio, dsp_params={
        ...     'threshold': 0.3,  # Aggressive clipping
        ...     'mix': 0.8        # 80% wet signal
        ... })

    Advanced Usage (shaping_mode=True):
        >>> # Create with waveshaping controls
        >>> dist = HardDist(sample_rate=44100, shaping_mode=True)
        >>> # Process with detailed control
        >>> output = dist(input_audio, dsp_params={
        ...     'threshold': 0.5,      # Moderate clipping
        ...     'pre_gain_db': 15.0,   # Drive into clipping
        ...     'post_gain_db': -12.0, # Compensate output
        ...     'dc_bias': 0.1,        # Add asymmetry
        ...     'mix': 0.7             # 70% wet
        ... })
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to hard clipping.
   
        Adds:
            threshold: Clipping threshold level (0.1 to 1.0)
        """
        self.params['threshold'] = EffectParam(min_val=0.1, max_val=1.0)
    
    def _apply_distortion(self, x, params):
        """Apply hard-clipping distortion to the input signal.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Must contain:
                - threshold: Clipping threshold level

        Returns:
            torch.Tensor: Distorted audio tensor of same shape as input
        """
        threshold = params['threshold'].unsqueeze(-1).unsqueeze(-1)
        return torch.clamp(x, min=-threshold, max=threshold)

class DoubleSoftDist(BaseDistortion):
    """Differentiable implementation of double soft-clipping distortion with asymmetric controls.

    Implementation is based on:

    ..  [4] https://jatinchowdhury18.medium.com/complex-nonlinearities-episode-1-double-soft-clipper-5ce826fa82d6
    
    
    This processor implements a sophisticated dual-stage soft-clipping distortion with independent
    control over positive and negative waveform shaping. It provides precise control over clipping
    characteristics, allowing creation of asymmetric distortion with variable slopes and limits.

    The transfer function is piecewise and applies separately to positive and negative regions:

    Positive region (x > 0):
    
    .. math::

        y = \\begin{cases} 
            upper\\_lim & x ≥ \\frac{1}{slope} \\\\
            \\frac{3}{2}upper\\_lim(slope⋅x - \\frac{(slope⋅x)^3}{3}) + \\frac{upper\\_lim}{2} & -\\frac{1}{slope} < x < \\frac{1}{slope} \\\\
            0 & x ≤ -\\frac{1}{slope}
        \\end{cases}

    Negative region (x ≤ 0):
    
    .. math::

        y = \\begin{cases} 
            0 & x ≥ \\frac{1}{slope} \\\\
            \\frac{3}{2}lower\\_lim(slope⋅x - \\frac{(slope⋅x)^3}{3}) + \\frac{lower\\_lim}{2} & -\\frac{1}{slope} < x < \\frac{1}{slope} \\\\
            lower\\_lim & x ≤ -\\frac{1}{slope}
        \\end{cases}

    Args:
    sample_rate (int): Audio sample rate in Hz
    shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        When True, enables:
        - Pre-gain control
        - DC bias adjustment
        - Post-gain control
        - Automatic DC filtering

    Parameters Details:
    Basic Mode (shaping_mode=False):
        upper_lim: Positive clipping limit
            - Range: 0.1 to 1.0
            - Controls maximum positive output
            
        lower_lim: Negative clipping limit
            - Range: -1.0 to -0.1
            - Controls maximum negative output
            
        slope: Transfer function steepness
            - Range: 1.0 to 10.0
            - Higher values create sharper transitions
            
        x_off_factor: Offset control
            - Range: 0.0 to 1.0
            - Affects symmetry of clipping curve
            
        upper_skew: Positive region shaping
            - Range: 0.1 to 2.0
            - Controls shape of positive overdrive
            
        lower_skew: Negative region shaping
            - Range: 0.1 to 2.0
            - Controls shape of negative overdrive
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only distorted signal

    Shaping Mode (shaping_mode=True):
        Adds standard shaping controls:
        pre_gain_db: Input gain before distortion
            - Range: -24.0 to 24.0 dB
            
        post_gain_db: Output gain after distortion
            - Range: -24.0 to 0.0 dB
            
        dc_bias: DC offset before distortion
            - Range: -0.2 to 0.2

    Examples:
    Basic Usage:
        >>> # Create a double soft clipper
        >>> dist = DoubleSoftDist(sample_rate=44100)
        >>> # Process with asymmetric settings
        >>> output = dist(input_audio, dsp_params={
        ...     'upper_lim': 0.8,
        ...     'lower_lim': -0.6,
        ...     'slope': 3.0,
        ...     'x_off_factor': 0.2,
        ...     'upper_skew': 1.5,
        ...     'lower_skew': 1.2,
        ...     'mix': 0.7
        ... })

    Advanced Usage (shaping_mode=True):
        >>> # Create with waveshaping controls
        >>> dist = DoubleSoftDist(sample_rate=44100, shaping_mode=True)
        >>> output = dist(input_audio, dsp_params={
        ...     'upper_lim': 0.8,
        ...     'lower_lim': -0.6,
        ...     'slope': 3.0,
        ...     'x_off_factor': 0.2,
        ...     'upper_skew': 1.5,
        ...     'lower_skew': 1.2,
        ...     'pre_gain_db': 12.0,
        ...     'post_gain_db': -6.0,
        ...     'dc_bias': 0.1,
        ...     'mix': 0.7
        ... })
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to double soft clipping.
   
        Adds:
            upper_lim: Positive clipping limit (0.1 to 1.0)
            lower_lim: Negative clipping limit (-1.0 to -0.1)
            slope: Transfer function steepness (1.0 to 10.0)
            x_off_factor: Offset control (0.0 to 1.0)
            upper_skew: Positive region shaping (0.1 to 2.0)
            lower_skew: Negative region shaping (0.1 to 2.0)
        """
        self.params.update({
            'upper_lim': EffectParam(min_val=0.1, max_val=1.0),
            'lower_lim': EffectParam(min_val=-1.0, max_val=-0.1),
            'slope': EffectParam(min_val=1.0, max_val=10.0),
            'x_off_factor': EffectParam(min_val=0.0, max_val=1.0),
            'upper_skew': EffectParam(min_val=0.1, max_val=2.0),
            'lower_skew': EffectParam(min_val=0.1, max_val=2.0),
        })
    
    def _apply_distortion(self, x, params):
        """Apply double soft-clipping distortion to the input signal.

        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Must contain:
                - upper_lim: Positive clipping limit
                - lower_lim: Negative clipping limit
                - slope: Transfer function steepness
                - x_off_factor: Offset control
                - upper_skew: Positive region shaping
                - lower_skew: Negative region shaping

        Returns:
            torch.Tensor: Distorted audio tensor of same shape as input
            
        Note:
            Processes positive and negative regions independently using
            different shaping parameters for each region.
        """
        # Get parameters and reshape for broadcasting
        # For shape [batch_size, 1, 1]
        upper_lim = params['upper_lim'].view(-1, 1, 1)
        lower_lim = params['lower_lim'].view(-1, 1, 1)
        slope = params['slope'].view(-1, 1, 1)
        x_off_factor = params['x_off_factor'].view(-1, 1, 1)
        upper_skew = params['upper_skew'].view(-1, 1, 1)
        lower_skew = params['lower_skew'].view(-1, 1, 1)
        
        # Calculate offset based on slope and x_off_factor
        x_off = (1/slope) * slope**x_off_factor
        
        # Initialize output tensor
        y = torch.zeros_like(x)
        
        # Create masks for positive and negative regions
        pos_mask = x > 0
        neg_mask = x <= 0
        
        # Process positive values
        if torch.any(pos_mask):
            # Apply offset and skew while preserving batch dimensions
            x_pos = (x * pos_mask - x_off) * upper_skew
            
            # Create masks for different regions
            pos_high = x_pos >= 1/slope
            pos_low = x_pos <= -1/slope
            pos_mid = ~pos_high & ~pos_low & pos_mask
            
            # Apply transfer function for each region
            y = torch.where(pos_high & pos_mask, upper_lim, y)
            y = torch.where(pos_low & pos_mask, torch.zeros_like(x), y)
            
            # Process middle region
            x_mid = x_pos * pos_mid
            mid_out = (3/2) * upper_lim * (slope*x_mid - (slope*x_mid)**3 / 3) / 2 + (upper_lim/2) * pos_mid
            y = torch.where(pos_mid, mid_out, y)
        
        # Process negative values
        if torch.any(neg_mask):
            # Apply offset and skew while preserving batch dimensions
            x_neg = (x * neg_mask + x_off) * lower_skew
            
            # Create masks for different regions
            neg_high = x_neg >= 1/slope
            neg_low = x_neg <= -1/slope
            neg_mid = ~neg_high & ~neg_low & neg_mask
            
            # Apply transfer function for each region
            y = torch.where(neg_high & neg_mask, torch.zeros_like(x), y)
            y = torch.where(neg_low & neg_mask, lower_lim, y)
            
            # Process middle region
            x_mid = x_neg * neg_mid
            mid_out = (3/2) * -lower_lim * (slope*x_mid - (slope*x_mid)**3 / 3) / 2 + (lower_lim/2) * neg_mid
            y = torch.where(neg_mid, mid_out, y)
        
        return y

class CubicDist(BaseDistortion):
    """Differentiable implementation of cubic distortion.

    This processor implements distortion using a cubic polynomial transfer function,
    creating asymmetric clipping characteristics by adding a scaled cubic term to the
    input signal. This approach generates both even and odd harmonics, providing a 
    rich timbral modification.

    The transfer function is:

    .. math::

        y = x + drive * x^3

    where:
    - x is the input signal
    - drive is the intensity control (derived from drive_db)
    - drive = 10^{drive\\_db/20}

    Args:
    sample_rate (int): Audio sample rate in Hz
    shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        When True, enables:
        - Pre-gain control
        - DC bias adjustment
        - Post-gain control
        - Automatic DC filtering

    Parameters Details:
    Basic Mode (shaping_mode=False):
        drive_db: Distortion intensity
            - Range: -24.0 to 24.0 dB
            - Controls amplitude of cubic term
            - Higher values create more distortion
            - Negative values reduce distortion
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only distorted signal

    Shaping Mode (shaping_mode=True):
        Adds:
        pre_gain_db: Input gain before distortion
            - Range: -24.0 to 24.0 dB
            - Controls overall drive level
            
        post_gain_db: Output gain after distortion
            - Range: -24.0 to 0.0 dB
            - Compensates for level changes
            
        dc_bias: DC offset before distortion
            - Range: -0.2 to 0.2
            - Affects harmonic balance

    Examples:
    Basic DSP Usage:
        >>> # Create a cubic distortion
        >>> dist = CubicDist(sample_rate=44100)
        >>> # Process with moderate drive
        >>> output = dist(input_audio, dsp_params={
        ...     'drive_db': 12.0,  # 12dB of drive
        ...     'mix': 0.7        # 70% wet signal
        ... })

    Advanced Usage (shaping_mode=True):
        >>> # Create with waveshaping controls
        >>> dist = CubicDist(sample_rate=44100, shaping_mode=True)
        >>> # Process with detailed control
        >>> output = dist(input_audio, dsp_params={
        ...     'drive_db': 18.0,      # Heavy distortion
        ...     'pre_gain_db': 6.0,    # Additional input drive
        ...     'post_gain_db': -12.0, # Output compensation
        ...     'dc_bias': 0.05,       # Slight asymmetry
        ...     'mix': 0.8             # 80% wet
        ... })
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to cubic distortion.
   
        Adds:
            drive_db: Distortion intensity (-24.0 to 24.0 dB)
        """
        self.params['drive_db'] = EffectParam(min_val=-24.0, max_val=24.0)
    
    def _apply_distortion(self, x, params):
        """Apply cubic distortion to the input signal.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Must contain:
                - drive_db: Distortion intensity in dB

        Returns:
            torch.Tensor: Distorted audio tensor of same shape as input
            
        Note:
            Drive is converted from dB to linear scaling before application
        """
        drive = params['drive_db'].view(-1, 1, 1)
        drive = 10 ** (drive / 20.0)
        x_dist = x + drive * x**3
        return x_dist
    
class RectifierDist(BaseDistortion):
    """Differentiable implementation of rectifier distortion.

    This processor implements half-wave and full-wave rectification with variable threshold,
    allowing smooth interpolation between rectification modes. The effect is similar to diode 
    clipping circuits, creating characteristic asymmetric distortion with rich harmonic content.

    The transfer function interpolates between:

    Half-wave (mode = 0):
    
    .. math::

        y = \\begin{cases} 
            x & x > threshold \\\\
            0 & |x| ≤ threshold \\\\
            0 & x < -threshold
        \\end{cases}

    Full-wave (mode = 1):
    
    .. math::

        y = \\begin{cases} 
            |x| & |x| > threshold \\\\
            0 & |x| ≤ threshold
        \\end{cases}

    Args:
    sample_rate (int): Audio sample rate in Hz
    shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        When True, enables:
        - Pre-gain control
        - DC bias adjustment
        - Post-gain control
        - Automatic DC filtering

    Parameters Details:
    Basic Mode (shaping_mode=False):
        mode: Rectification type
            - Range: 0.0 to 1.0
            - 0.0: Half-wave rectification
            - 1.0: Full-wave rectification
            - Intermediate values blend between modes
            
        threshold: Signal threshold for rectification
            - Range: 0.0 to 1.0
            - Signals below threshold are set to zero
            - Higher values create gating effects
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only rectified signal

    Shaping Mode (shaping_mode=True):
        Adds:
        pre_gain_db: Input gain before rectification
            - Range: -24.0 to 24.0 dB
            - Controls amount of signal above threshold
            
        post_gain_db: Output gain after rectification
            - Range: -24.0 to 0.0 dB
            - Compensates for level changes
            
        dc_bias: DC offset before rectification
            - Range: -0.2 to 0.2
            - Affects rectification symmetry

    Note:
    - Half-wave creates strong asymmetric distortion
    - Full-wave doubles frequency content
    - Threshold creates gating effects
    - Useful for extreme timbral modification
    - Can generate octave-up effects (full-wave)

    Examples:
    Basic DSP Usage:
        >>> # Create a rectifier distortion
        >>> dist = RectifierDist(sample_rate=44100)
        >>> # Process with half-wave rectification
        >>> output = dist(input_audio, dsp_params={
        ...     'mode': 0.0,       # Half-wave mode
        ...     'threshold': 0.2,  # Low threshold
        ...     'mix': 0.6         # 60% wet signal
        ... })

    Advanced Usage (shaping_mode=True):
        >>> # Create with waveshaping controls
        >>> dist = RectifierDist(sample_rate=44100, shaping_mode=True)
        >>> # Process with detailed control
        >>> output = dist(input_audio, dsp_params={
        ...     'mode': 0.7,          # Blend of half/full wave
        ...     'threshold': 0.3,     # Moderate threshold
        ...     'pre_gain_db': 12.0,  # Drive into rectification
        ...     'post_gain_db': -6.0, # Compensate output
        ...     'dc_bias': 0.1,       # Add asymmetry
        ...     'mix': 0.8            # 80% wet
        ... })
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to rectifier distortion.
   
        Adds:
            mode: Rectification type (0.0 to 1.0)
            threshold: Signal threshold (0.0 to 1.0)
        """
        self.params.update({
            'mode': EffectParam(min_val=0.0, max_val=1.0),  # 0: half-wave, 1: full-wave
            'threshold': EffectParam(min_val=0.0, max_val=1.0)  # threshold for rectification
        })
    
    def _apply_distortion(self, x, params):
        """Apply rectification distortion to the input signal.

        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Must contain:
                - mode: Rectification type (0: half-wave, 1: full-wave)
                - threshold: Signal threshold for rectification

        Returns:
            torch.Tensor: Rectified audio tensor of same shape as input
            
        Note:
            Interpolates smoothly between half-wave and full-wave rectification
            using the mode parameter.
        """
        # Get parameters
        mode = params['mode'].view(-1, 1, 1)
        threshold = params['threshold'].view(-1, 1, 1)
        
        # Apply threshold
        x = torch.where(torch.abs(x) < threshold, torch.zeros_like(x), x)
        
        # Interpolate between half and full wave rectification
        half_wave = torch.relu(x)  # half-wave rectification
        full_wave = torch.abs(x)   # full-wave rectification
        
        return torch.lerp(half_wave, full_wave, mode)
    
class ArcTanDist(BaseDistortion):
    """Differentiable implementation of arctangent distortion.

    This processor implements smooth distortion using the arctangent function for waveshaping.
    Similar to tanh distortion but with slightly different harmonic characteristics, providing
    musical saturation with natural compression and rich overtones.

    The transfer function is:

    .. math::

        y = \\frac{2}{\\pi} \\arctan(x * drive)

    where:
    - x is the input signal
    - drive controls distortion intensity
    - 2/π factor normalizes output to [-1, 1] range

    Args:
    sample_rate (int): Audio sample rate in Hz
    shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        When True, enables:
        - Pre-gain control
        - DC bias adjustment
        - Post-gain control
        - Automatic DC filtering

    Parameters Details:
    Basic Mode (shaping_mode=False):
        drive: Distortion intensity
            - Range: 0.1 to 10.0
            - Controls slope of arctangent curve
            - Higher values create more saturation
            - Lower values provide subtle warming
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only distorted signal

    Shaping Mode (shaping_mode=True):
        Adds:
        pre_gain_db: Input gain before distortion
            - Range: -24.0 to 24.0 dB
            - Controls signal level into arctangent
            
        post_gain_db: Output gain after distortion
            - Range: -24.0 to 0.0 dB
            - Compensates for level changes
            
        dc_bias: DC offset before distortion
            - Range: -0.2 to 0.2
            - Affects harmonic content

    Note:
    - Creates smooth, musical distortion
    - Natural compression at high drive levels
    - Generates primarily odd harmonics
    - Similar to but smoother than tanh
    - Output automatically normalized to [-1, 1]

    Examples:
    Basic DSP Usage:
        >>> # Create an arctangent distortion
        >>> dist = ArcTanDist(sample_rate=44100)
        >>> # Process with moderate drive
        >>> output = dist(input_audio, dsp_params={
        ...     'drive': 4.0,  # Moderate saturation
        ...     'mix': 0.7    # 70% wet signal
        ... })

    Advanced Usage (shaping_mode=True):
        >>> # Create with waveshaping controls
        >>> dist = ArcTanDist(sample_rate=44100, shaping_mode=True)
        >>> # Process with detailed control
        >>> output = dist(input_audio, dsp_params={
        ...     'drive': 6.0,         # Strong saturation
        ...     'pre_gain_db': 6.0,   # Additional drive
        ...     'post_gain_db': -3.0, # Slight attenuation
        ...     'dc_bias': 0.05,      # Slight asymmetry
        ...     'mix': 0.8            # 80% wet
        ... })
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to arctangent distortion.
   
        Adds:
            drive: Distortion intensity (0.1 to 10.0)
        """
        self.params.update({
            'drive': EffectParam(min_val=0.1, max_val=10.0)  # Drive amount for atan curve
        })
    
    def _apply_distortion(self, x, params):
        """Apply arctangent distortion to the input signal.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Must contain:
                - drive: Distortion intensity

        Returns:
            torch.Tensor: Distorted audio tensor of same shape as input
            
        Note:
            Output is automatically normalized to [-1, 1] range using
            the (2/π) scaling factor.
        """
        # Get drive parameter
        drive = params['drive'].view(-1, 1, 1)
        
        # Apply arctangent with scaling
        # (2/π) factor normalizes output to [-1, 1] range
        return (2/np.pi) * torch.atan(x * drive)
    
class ExponentialDist(BaseDistortion):
    """Differentiable implementation of exponential distortion with asymmetry control.

    This processor implements distortion using exponential curves, featuring independent control
    over positive and negative regions. It creates dynamic saturation characteristics with
    natural compression and adjustable asymmetry for rich harmonic content.

    The transfer function is:

    .. math::

        y = sign(x) * (1 - e^{-|x| * drive * A(x)})

    where:
    - x is the input signal
    - drive controls overall distortion intensity
    - A(x) is the asymmetry function:
        - A(x) = 1 for x ≥ 0
        - A(x) = asymmetry for x < 0

    Args:
    sample_rate (int): Audio sample rate in Hz
    shaping_mode (bool): Whether to enable waveshaping controls. Defaults to False.
        When True, enables:
        - Pre-gain control
        - DC bias adjustment
        - Post-gain control
        - Automatic DC filtering

    Parameters Details:
    Basic Mode (shaping_mode=False):
        drive: Distortion intensity
            - Range: 0.1 to 10.0
            - Controls steepness of exponential curve
            - Higher values create more saturation
            - Affects both positive and negative regions
            
        asymmetry: Positive/negative balance
            - Range: 0.1 to 2.0
            - 1.0: Symmetric distortion
            - <1.0: More negative distortion
            - >1.0: More positive distortion
            
        mix: Wet/dry mix ratio
            - Range: 0.0 to 1.0
            - 0.0: Only clean signal
            - 1.0: Only distorted signal

    Shaping Mode (shaping_mode=True):
        Adds:
        pre_gain_db: Input gain before distortion
            - Range: -24.0 to 24.0 dB
            - Controls signal level into exponential curve
            
        post_gain_db: Output gain after distortion
            - Range: -24.0 to 0.0 dB
            - Compensates for level changes
            
        dc_bias: DC offset before distortion
            - Range: -0.2 to 0.2
            - Affects harmonic balance

    Note:
    - Creates smooth compression curve
    - Asymmetry control for harmonic shaping
    - Natural limiting at high drive levels
    - Output automatically bounded to [-1, 1]
    - Particularly effective on transient material

    Examples:
    Basic DSP Usage:
        >>> # Create an exponential distortion
        >>> dist = ExponentialDist(sample_rate=44100)
        >>> # Process with asymmetric settings
        >>> output = dist(input_audio, dsp_params={
        ...     'drive': 3.0,      # Moderate drive
        ...     'asymmetry': 1.5,  # More positive distortion
        ...     'mix': 0.7         # 70% wet signal
        ... })

    Advanced Usage (shaping_mode=True):
        >>> # Create with waveshaping controls
        >>> dist = ExponentialDist(sample_rate=44100, shaping_mode=True)
        >>> # Process with detailed control
        >>> output = dist(input_audio, dsp_params={
        ...     'drive': 5.0,         # Strong drive
        ...     'asymmetry': 0.8,     # More negative distortion
        ...     'pre_gain_db': 6.0,   # Additional input drive
        ...     'post_gain_db': -6.0, # Output attenuation
        ...     'dc_bias': 0.1,       # Slight positive offset
        ...     'mix': 0.8            # 80% wet
        ... })
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to exponential distortion.
   
        Adds:
            drive: Distortion intensity (0.1 to 10.0)
            asymmetry: Positive/negative balance (0.1 to 2.0)
        """
        self.params.update({
            'drive': EffectParam(min_val=0.1, max_val=10.0),  # Drive amount for exp curve
            'asymmetry': EffectParam(min_val=0.1, max_val=2.0)  # Controls positive/negative asymmetry
        })
    
    def _apply_distortion(self, x, params):
        """Apply exponential distortion to the input signal.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Must contain:
                - drive: Distortion intensity
                - asymmetry: Positive/negative balance

        Returns:
            torch.Tensor: Distorted audio tensor of same shape as input
            
        Note:
            Processes positive and negative regions independently using
            the asymmetry parameter to control their relative intensity.
        """
        # Get parameters
        drive = params['drive'].view(-1, 1, 1)
        asymmetry = params['asymmetry'].view(-1, 1, 1)
        
        # Split processing for positive and negative values
        # Use sign(x) to maintain the sign while applying different curves
        return torch.sign(x) * (1 - torch.exp(-torch.abs(x) * drive * torch.where(x >= 0, torch.ones_like(x), asymmetry)))