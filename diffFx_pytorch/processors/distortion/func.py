import torch 
import numpy as np 
from typing import Dict, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..filters import DCFilter


class BaseDistortion(ProcessorsBase):
    """Base class for implementing various distortion effects with optional waveshaping controls.

    This class provides a foundation for implementing different types of distortion effects,
    with optional waveshaping parameters for more detailed control over the distortion characteristics.
    It includes pre/post gain staging, DC bias control, and automatic DC filtering when in shaping mode.
    """
    def __init__(self, sample_rate, param_range=None,shaping_mode=False):
        """Initialize the distortion processor."""
        self.shaping_mode = shaping_mode
        super().__init__(sample_rate, param_range)
        
        if self.shaping_mode:
            self.dc_filter = DCFilter(sample_rate, learnable=False)
    
    def _register_default_parameters(self):
        """Register default parameters for the distortion processor."""
        base_params = {
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        }
        
        shaping_params = {
            'pre_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
            'post_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
            'dc_bias': EffectParam(min_val=-0.2, max_val=0.2),
        }
        
        if self.shaping_mode:
            self.params = {**base_params, **shaping_params}
        else:
            self.params = base_params
            
        # Add any additional parameters specific to the distortion type
        self._add_specific_parameters()
    
    def _add_specific_parameters(self):
        pass
    
    def _apply_distortion(self, x):
        """Apply the distortion transfer function to the input signal."""
        raise NotImplementedError
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None]=None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
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
        
        mix = params['mix']
        
        if self.shaping_mode:
            pre_gain = 10 ** (params['pre_gain_db'] / 20.0)
            post_gain = 10 ** (params['post_gain_db'] / 20.0)
            pre_gain = pre_gain.unsqueeze(-1).unsqueeze(-1)
            post_gain = post_gain.unsqueeze(-1).unsqueeze(-1)
            dc_bias = params['dc_bias'].unsqueeze(-1).unsqueeze(-1)
            x_driven = pre_gain * x + dc_bias
        else:
            x_driven = x
        
        x_distorted = self._apply_distortion(x_driven, params)
        
        if self.shaping_mode:
            x_processed = post_gain * x_distorted
            x_processed = self.dc_filter(x_processed, None, None)
        else:
            x_processed = x_distorted
        
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
    """
    def _apply_distortion(self, x, params):
        """Apply soft-clipping distortion to the input signal.
   
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            params (dict): Processing parameters (not used in this implementation)
            
        Returns:
            torch.Tensor: Distorted audio tensor of same shape as input
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
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to double soft clipping."""
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
        """
        upper_lim = params['upper_lim'].view(-1, 1, 1)
        lower_lim = params['lower_lim'].view(-1, 1, 1)
        slope = params['slope'].view(-1, 1, 1)
        x_off_factor = params['x_off_factor'].view(-1, 1, 1)
        upper_skew = params['upper_skew'].view(-1, 1, 1)
        lower_skew = params['lower_skew'].view(-1, 1, 1)
        
        x_off = (1/slope) * slope**x_off_factor
        
        y = torch.zeros_like(x)
        
        pos_mask = x > 0
        neg_mask = x <= 0
        
        if torch.any(pos_mask):
            x_pos = (x * pos_mask - x_off) * upper_skew
            
            pos_high = x_pos >= 1/slope
            pos_low = x_pos <= -1/slope
            pos_mid = ~pos_high & ~pos_low & pos_mask
            
            y = torch.where(pos_high & pos_mask, upper_lim, y)
            y = torch.where(pos_low & pos_mask, torch.zeros_like(x), y)
            
            x_mid = x_pos * pos_mid
            mid_out = (3/2) * upper_lim * (slope*x_mid - (slope*x_mid)**3 / 3) / 2 + (upper_lim/2) * pos_mid
            y = torch.where(pos_mid, mid_out, y)
        
        if torch.any(neg_mask):
            x_neg = (x * neg_mask + x_off) * lower_skew
            
            neg_high = x_neg >= 1/slope
            neg_low = x_neg <= -1/slope
            neg_mid = ~neg_high & ~neg_low & neg_mask
            
            y = torch.where(neg_high & neg_mask, torch.zeros_like(x), y)
            y = torch.where(neg_low & neg_mask, lower_lim, y)
            
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
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to rectifier distortion."""
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
        """
        mode = params['mode'].view(-1, 1, 1)
        threshold = params['threshold'].view(-1, 1, 1)
        
        x = torch.where(torch.abs(x) < threshold, torch.zeros_like(x), x)
        
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
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to arctangent distortion."""
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
        """
        drive = params['drive'].view(-1, 1, 1)
        
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
    """
    def _add_specific_parameters(self):
        """Register additional parameters specific to exponential distortion."""
        self.params.update({
            'drive': EffectParam(min_val=0.1, max_val=10.0), 
            'asymmetry': EffectParam(min_val=0.1, max_val=2.0) 
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
        """
        drive = params['drive'].view(-1, 1, 1)
        asymmetry = params['asymmetry'].view(-1, 1, 1)
        return torch.sign(x) * (1 - torch.exp(-torch.abs(x) * drive * torch.where(x >= 0, torch.ones_like(x), asymmetry)))
    
