import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.envelope import TruncatedOnePoleIIRFilter, Ballistics
from ..core.utils import ms_to_z_alpha
from ..filters import LinkwitzRileyFilter


class TransientShaper(ProcessorsBase):
    """Differentiable implementation of a transient shaping processor.
    
    The implementation is based on following: 
    
    ..  [1] SPL Transient Shaper Design Manual: 
            https://spl.audio/wp-content/uploads/transient_designer_2_9946_manual.pdf
    ..  [2] https://www.elysia.com/transient-designer-story/
    ..  [3] https://github.com/sevagh/multiband-transient-shaper
     
    This processor manipulates the dynamic characteristics of audio signals by detecting 
    and modifying transients using dual envelope followers. It operates by comparing 
    the difference between fast and slow envelope followers applied to the signal's 
    power envelope derivative, allowing it to either enhance (attack mode) or soften 
    (sustain mode) transient content.

    The processing chain follows these steps:
    1. Power envelope extraction using a one-pole IIR filter
    2. Power envelope derivative calculation
    3. Dual envelope following with separate fast and slow time constants
    4. Attack gain curve generation from envelope difference
    5. Gain application based on mode selection

    The power envelope is computed using a one-pole IIR filter:

    .. math::

        P(n) = (1-\\alpha_{power}) \cdot x^2(n) + \\alpha_{power} \cdot P(n-1)

    The envelope followers use different time constants for attack and release:

    .. math::

        y(n) = \\begin{cases} 
        (1-\\alpha_{attack}) \cdot x(n) + \\alpha_{attack} \cdot y(n-1),  & x(n) ≥ y(n-1) \\\\
        (1-\\alpha_{release}) \cdot x(n) + \\alpha_{release} \cdot y(n-1), & x(n) < y(n-1)
        \\end{cases}

    where:
        - x(n) is the power envelope derivative
        - y(n) is the envelope follower output
        - \\alpha values are computed as: \\alpha = exp(-1/(sample_rate * time_ms/1000))

    The final gain curve is computed as:

    
    .. math::
    
        g(n) = \\begin{cases}
        E_{fast}(n) - E_{slow}(n), & \\text{attack mode} \\\\
        1 - (E_{fast}(n) - E_{slow}(n)), & \\text{sustain mode}
        \\end{cases}

    Args:
        mode (str, optional): Processing mode, either "attack" or "sustain". Defaults to "attack".
        sample_rate (int, optional): Audio sample rate in Hz. Defaults to 44100.

    Parameters Details:
        power_mem_ms: Power envelope memory time in milliseconds
            - Range: 0.1 to 5.0 ms
            - Controls smoothing of the input signal's power envelope
            - Shorter times provide faster response to transients
            
        fast_attack_ms: Fast envelope follower attack time
            - Range: 0.01 to 5.0 ms
            - Controls detection of rapid transients
            - Very short times catch quick attacks
            
        slow_attack_ms: Slow envelope follower attack time
            - Range: 10.0 to 50.0 ms
            - Controls the sustained energy detection
            - Longer times smooth out transient detection
            
        release_ms: Release time for both envelope followers
            - Range: 10.0 to 100.0 ms
            - Controls recovery time after transients
            - Affects how quickly the processor resets

    Examples:
        Basic DSP Usage:
            >>> # Create a transient shaper for attack enhancement
            >>> shaper = TransientShaper(mode="attack", sample_rate=44100)
            >>> # Process audio with typical drum settings
            >>> output = shaper(input_audio, dsp_params={
            ...     'power_mem_ms': 1.0,    # Quick power envelope
            ...     'fast_attack_ms': 0.1,  # Catch fast transients
            ...     'slow_attack_ms': 20.0, # Smooth sustained content
            ...     'release_ms': 50.0      # Natural decay
            ... })

        Neural Network Control:
            >>> # 1. Parameter prediction network
            >>> class TransientController(nn.Module):
            ...     def __init__(self, input_size):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 64),
            ...             nn.ReLU(),
            ...             nn.Linear(64, 4),  # 4 parameters
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Process with features
            >>> controller = TransientController(input_size=16)
            >>> features = torch.randn(batch_size, 16)
            >>> norm_params = controller(features)
            >>> output = shaper(input_audio, norm_params=norm_params)
    """
    def __init__(self, mode="attack", sample_rate=44100):
        super().__init__(sample_rate=sample_rate)
        self.mode = mode
        self.ballistics = Ballistics()
        self.power_filter = TruncatedOnePoleIIRFilter(iir_len=16384)
        
    def _register_default_parameters(self):
        """Register default parameters with their ranges.
        
        Registers four main parameters:
        - Power envelope memory time
        - Fast attack time constant
        - Slow attack time constant
        - Release time constant
        """
        self.params = {
            'power_mem_ms': EffectParam(min_val=0.1, max_val=5.0),
            'fast_attack_ms': EffectParam(min_val=0.01, max_val=5.0),
            'slow_attack_ms': EffectParam(min_val=10.0, max_val=50.0),
            'release_ms': EffectParam(min_val=10.0, max_val=100.0)
        }
                
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        
        """Process input audio through the transient shaper.
        
        Implementation follows these steps:
        1. Input normalization
        2. Power envelope extraction using one-pole IIR filter
        3. Power derivative calculation
        4. Dual envelope following with different attack times
        5. Attack gain curve generation
        6. Final gain application based on mode
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify the following parameters:
                - power_mem_ms: Power envelope memory time (ms)
                - fast_attack_ms: Fast envelope attack time (ms)
                - slow_attack_ms: Slow envelope attack time (ms)
                - release_ms: Release time (ms)
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.
        
        Returns:
            torch.Tensor: Processed audio tensor with enhanced or softened transients,
                same shape as input.
                
        Note:
            The implementation uses a dual-envelope approach where the difference
            between fast and slow envelope followers creates a control signal
            that modulates transients. In "attack" mode, positive differences
            enhance transients, while in "sustain" mode, they are softened.
        """
        bs, chs, t = x.shape 
        if chs == 2:
            raise ValueError("Input to the transient shaper should be mono")
        
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
                
        # 1. Normalize
        x_peak = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        x_norm = x / (x_peak + 1e-8)

        # 2. Get coefficients and ensure proper shapes
        z_alpha_power = ms_to_z_alpha(params['power_mem_ms'], self.sample_rate).unsqueeze(-1)  # Shape: (batch, 1)
        g_fast = params['fast_attack_ms'] #self._ms_to_coeff(params['fast_attack_ms'])#.unsqueeze(-1)         # Shape: (batch, 1)
        g_slow = params['slow_attack_ms'] #self._ms_to_coeff(params['slow_attack_ms'])#.unsqueeze(-1)         # Shape: (batch, 1)
        g_release = params['release_ms'] #self._ms_to_coeff(params['release_ms'])#.unsqueeze(-1)          # Shape: (batch, 1)
        
        # Power Envelope 
        # x_squared = x_norm.square().mean(-2)  # Shape: (batch, time)
        x_squared = x_norm.squeeze(1)
        
        # Ensure power filter input shapes are correct
        power = self.power_filter(x_squared, z_alpha_power)  # Shape: (batch, time)
        
        # Compute power derivative
        power_deriv = torch.zeros_like(power)
        power_deriv[:, 0] = power[:, 0]
        power_deriv[:, 1:] = power[:, 1:] - power[:, :-1]
        
        # 3. Ballistics processing for fast and slow envelopes
        z_alpha_fast = torch.stack([
            ms_to_z_alpha(g_release, self.sample_rate),
            ms_to_z_alpha(g_fast, self.sample_rate),
            #ms_to_z_alpha(g_release, self.sample_rate),
        ], dim=-1)  # Shape: (batch, 2)
        
        z_alpha_slow = torch.stack([
            ms_to_z_alpha(g_release, self.sample_rate),
            ms_to_z_alpha(g_slow, self.sample_rate),
            #ms_to_z_alpha(g_release, self.sample_rate),
        ], dim=-1)  # Shape: (batch, 2)
        
        fast_env = self.ballistics(power_deriv, z_alpha_fast)
        slow_env = self.ballistics(power_deriv, z_alpha_slow)
        
        # 4. Attack gain curve
        attack_gain = fast_env - slow_env
        attack_gain = attack_gain / (torch.max(torch.abs(attack_gain), dim=-1, keepdim=True)[0] + 1e-8)
        
        # 5. Apply gain
        if self.mode == "attack":
            y = x_norm * attack_gain.unsqueeze(1)
        else:
            y = x_norm * (1.0 - attack_gain).unsqueeze(1)
        
        
        return y * x_peak 
