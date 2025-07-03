import torch 
from typing import Dict, Union
from ..base import EffectParam
from ..base_utils import check_params
from ..core.envelope import Ballistics
from ..core.utils import ms_to_alpha
from .expander import Expander

# Noise Gate  
class NoiseGate(Expander):
    """Differentiable feedforward noise gate.
    
    This processor implements a feedforward noise gate with configurable knee curves and envelope following methods.
    The noise gate reduces the level of signals that fall below the threshold, effectively removing background noise
    and unwanted low-level signals.

    A noise gate is essentially an extreme case of an expander with a very high ratio (typically 20:1 or higher).
    It provides more aggressive attenuation than a standard expander.

    Implementation is based on the following papers: 
    
    ..  [1] Giannoulis, Dimitrios, Michael Massberg, and Joshua D. Reiss. 
            "Digital dynamic range compressor design—A tutorial and analysis." 
            Journal of the Audio Engineering Society 60.6 (2012): 399-408.
    ..  [2] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    
    Processing Chain:
        1. Level Detection: RMS energy → dB
        2. Envelope Following: Attack/release smoothing
        3. Gain Computation: Apply gating curve
        4. Gain Application: Convert to linear gain and apply

    The noise gate implements the following gain computation:

    Hard Knee:
        .. math::

            g(x) = \\begin{cases}
                (T + \\frac{x - T}{R}) - x, & \\text{if } x \\leq T \\\\
                0, & \\text{if } x > T
            \\end{cases}

    Quadratic Knee:
        .. math::

            g(x) = \\begin{cases}
                (T + \\frac{x - T}{R}) - x, & \\text{if } x \\leq T - W/2 \\\\
                (\\frac{1}{R} - 1)\\frac{(x - T + W/2)^2}{4W}, & \\text{if } T - W/2 < x \\leq T + W/2 \\\\
                0, & \\text{if } x > T + W/2
            \\end{cases}

    Exponential Knee:
        .. math::

            g(x) = (\\frac{1}{R} - 1) \\frac{\\text{softplus}(k(T - x))}{k}

        where k = exp(W) is the knee factor
    
    Variables:
        - x: input level in dB
        - T: threshold in dB
        - R: ratio (typically 20:1 or higher for noise gates)
        - W: knee width in dB

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        param_range (Dict[str, EffectParam], optional): Parameter ranges.
        knee_type (str, optional): Type of gating knee curve. 
            Must be one of: "hard", "quadratic", "exponential". Defaults to "hard".
        smooth_type (str, optional): Type of envelope follower.
            Must be one of: "ballistics", "iir". Defaults to "ballistics".

    Attributes:
        knee_type (str): Current knee characteristic type
        smoothing_type (str): Current envelope follower type
        ballistics (Ballistics): Envelope follower for attack/release
        iir_filter (TruncatedOnePoleIIRFilter): IIR filter for smoothing

    Parameters Details:
        threshold_db: Level at which gating begins
            - Controls where gating starts to take effect
            - More negative values gate more of the signal
        ratio: Amount of attenuation below threshold
            - Higher ratios mean more aggressive gating
            - Typical values: 20:1 to 100:1 for noise gates
            - 1:1 means no gating
        knee_db: Width of the transition region around threshold
            - 0 dB creates a hard knee (sudden transition)
            - Larger values create smoother transitions
            - Affects how gradually gating is applied
        attack_ms: Time taken to react to increases in level
            - Shorter times mean faster response to transients
            - Longer times let more of the transient through
        release_ms: Time taken to react to decreases in level
            - Controls how quickly gating is reduced
            - Affects the "movement" of the gating

    Note:
        The processor supports the following parameter ranges:
            - threshold_db: Threshold level in dB (-90 to -20)
            - ratio: Gating ratio (1 to 100)
            - knee_db: Knee width in dB (0 to 6)
            - attack_ms: Attack time in ms (0.1 to 20)
            - release_ms: Release time in ms (5 to 1000)

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their DSP ranges
            - Ensure your network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()
        
    Examples
    --------
    Basic DSP Usage:
        >>> # Create a noise gate with hard knee
        >>> gate = NoiseGate(
        ...     sample_rate=44100,
        ...     knee_type="hard",
        ...     smooth_type="ballistics"
        ... )
        >>> # Process audio with dsp parameters
        >>> output = gate(input_audio, dsp_params={
        ...     'threshold_db': -40.0,
        ...     'ratio': 50.0,  # 50:1 ratio for aggressive gating
        ...     'knee_db': 0.0,
        ...     'attack_ms': 1.0,
        ...     'release_ms': 50.0
        ... })

    Neural Network Control:
        >>> # 1. Simple parameter prediction
        >>> class NoiseGateController(nn.Module):
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
        >>> num_params = gate.count_num_parameters()  # 5 parameters
        >>> controller = NoiseGateController(input_size=16, num_params=num_params)
        >>> 
        >>> # Process with features
        >>> features = torch.randn(batch_size, 16)  # Audio features
        >>> norm_params = controller(features)
        >>> output = gate(input_audio, norm_params=norm_params)
    """
    def __init__(self, sample_rate=44100, param_range=None):
        """Initialize the noise gate.

        Args:
            sample_rate (int): Audio sample rate in Hz
            param_range (Dict[str, EffectParam], optional): Parameter ranges.
            knee_type (str): Type of gating knee curve
            smooth_type (str): Type of envelope follower
        
        Raises:
            ValueError: If knee_type or smooth_type is invalid
        """
        super().__init__(sample_rate, param_range)
        
        self.knee_type = "hard"
        self.smooth_filter = Ballistics() # for smoothing 
        
    def _register_default_parameters(self):
        """Register default parameter ranges for the noise gate.
        """
        self.params = {
            'threshold_db': EffectParam(min_val=-90.0, max_val=-20.0),
            'ratio': EffectParam(min_val=1.0, max_val=100.0),  # High ratios for aggressive gating
            'knee_db': EffectParam(min_val=0.0, max_val=6.0),
            'attack_ms': EffectParam(min_val=0.1, max_val=20.0),  
            'release_ms': EffectParam(min_val=5.0, max_val=1000.0)
        }

    def _compute_gain(self, level_db: torch.Tensor, threshold_db: torch.Tensor,
                     ratio: torch.Tensor, knee_db: torch.Tensor) -> torch.Tensor:
        """Compute noise gate gain based on knee type.
        
        A noise gate is an extreme expander that applies aggressive attenuation
        below the threshold. The gain computation follows the same mathematical
        form as expansion but with higher ratios.
        
        Args:
            level_db (torch.Tensor): Input level in dB. Shape: (batch, time)
            threshold_db (torch.Tensor): Threshold in dB. Shape: (batch,)
            ratio (torch.Tensor): Gating ratio. Shape: (batch,)
            knee_db (torch.Tensor): Knee width in dB. Shape: (batch,)
            
        Returns:
            torch.Tensor: Gain reduction in dB. Shape: (batch, time)
            
        Note:
            The gain computation depends on the knee_type:
            - "hard": Sharp transition at threshold
            - "quadratic": Smooth quadratic transition around threshold
            - "exponential": Continuous transition using softplus
        """
        threshold_db = threshold_db.unsqueeze(-1)  # Shape: (batch, 1)
        ratio = ratio.unsqueeze(-1)  # Shape: (batch, 1)
        knee_db = knee_db.unsqueeze(-1)  # Shape: (batch, 1)
        
        below_thresh = level_db < threshold_db
        gain_db = torch.where(
            below_thresh,
            (1 - 1 / ratio) * (level_db - threshold_db),
            torch.zeros_like(level_db)
        )
        
        return gain_db

    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """Process input signal through the noise gate.

        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Dictionary with keys:
                - 'threshold_db': Level at which gating begins (-90 to -20 dB)
                - 'ratio': Amount of attenuation below threshold (1 to 100)
                - 'knee_db': Width of transition region around threshold (0 to 6 dB)
                - 'attack_ms': Time to react to level increases (0.1 to 20 ms)
                - 'release_ms': Time to react to level decreases (5 to 1000 ms)
                Each value should be a tensor of shape (batch_size,)
                Values will be mapped to their respective ranges
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
        """
        return super().process(x, norm_params, dsp_params)
            
        

# # Multiband Noise Gate 
# class MultiBandNoiseGate(MultiBandCompressor):
#     """Differentiable multi-band noise gate.
    
#     This processor splits the input signal into multiple frequency bands using
#     Linkwitz-Riley crossover filters and applies independent noise gating to each band.
#     The processed bands are then summed to produce the final output.

#     A noise gate is essentially an extreme case of an expander with a very high ratio
#     (typically 20:1 or higher). It provides more aggressive attenuation than a standard expander.

#     Implementation is based on the following references:
    
#     ..  [1] MATLAB Audio Toolbox. "Multiband Dynamic Range Compression."
#             https://www.mathworks.com/help/audio/ug/multiband-dynamic-range-compression.html
#     ..  [2] Giannoulis, D., Massberg, M., & Reiss, J. D. (2012).
#             "Digital dynamic range compressor design—A tutorial and analysis."
#             Journal of the Audio Engineering Society, 60(6), 399-408.
    
#     Processing Chain:
#         1. Band Splitting: Split input into frequency bands using Linkwitz-Riley filters
#         2. Per-band Processing:
#             a. Level Detection: Compute RMS energy and convert to dB
#             b. Envelope Following: Smooth level using attack/release ballistics
#             c. Gain Computation: Apply gating curve based on knee type
#             d. Gain Application: Convert to linear gain and apply to band
#         3. Band Summation: Sum all processed bands to create final output

#     Args:
#         sample_rate (int): Audio sample rate in Hz
#         param_range (Dict[str, EffectParam], optional): Parameter ranges.
#         num_bands (int, optional): Number of frequency bands. Defaults to 3.
#         knee_type (str, optional): Type of gating knee curve. 
#             Must be one of: "hard", "quadratic", "exponential". Defaults to "hard".
#         smooth_type (str, optional): Type of envelope follower.
#             Must be one of: "ballistics", "iir". Defaults to "ballistics".

#     Parameters Details:
#         For each band i (0 to num_bands-1):
#             band{i}_threshold_db: Level at which gating begins
#                 - Controls where gating starts for this band (-90 to -20 dB)
#             band{i}_ratio: Amount of attenuation below threshold
#                 - Higher ratios mean more aggressive gating (1 to 100)
#                 - Typical values: 20:1 to 100:1 for noise gates
#             band{i}_knee_db: Width of transition region around threshold
#                 - Controls how gradually gating is applied (0 to 6 dB)
#             band{i}_attack_ms: Time to react to level increases
#                 - Controls response to transients (0.1 to 20 ms)
#             band{i}_release_ms: Time to react to level decreases
#                 - Controls recovery time (5 to 1000 ms)
        
#         Crossover frequencies:
#             crossover{i}_freq: Split frequency between bands i and i+1
#                 - Logarithmically spaced between bands
#                 - Range varies based on position (20 Hz to 20 kHz)

#     Note:
#         - Uses 4th-order Linkwitz-Riley crossover filters
#         - Band splitting is done in series for proper phase alignment
#         - Each band has independent gating parameters
#         - Parameters can be controlled via normalized (0-1) or DSP values
#         - Total parameters = num_bands * 5 + (num_bands - 1)

#     Examples:
#         Basic Usage:
#             >>> # Create a 3-band noise gate
#             >>> mb_gate = MultiBandNoiseGate(
#             ...     sample_rate=44100,
#             ...     num_bands=3
#             ... )
#             >>> # Process with DSP parameters
#             >>> output = mb_gate(input_audio, dsp_params={
#             ...     'band0_threshold_db': -50.0,  # Low band
#             ...     'band0_ratio': 50.0,  # 50:1 ratio for aggressive gating
#             ...     'band0_knee_db': 0.0,
#             ...     'band0_attack_ms': 1.0,
#             ...     'band0_release_ms': 50.0,
#             ...     'band1_threshold_db': -45.0,  # Mid band
#             ...     'band1_ratio': 40.0,  # 40:1 ratio
#             ...     'band1_knee_db': 0.0,
#             ...     'band1_attack_ms': 0.5,
#             ...     'band1_release_ms': 30.0,
#             ...     'band2_threshold_db': -40.0,  # High band
#             ...     'band2_ratio': 30.0,  # 30:1 ratio
#             ...     'band2_knee_db': 0.0,
#             ...     'band2_attack_ms': 0.1,
#             ...     'band2_release_ms': 20.0,
#             ...     'crossover0_freq': 200.0,     # Low-Mid split
#             ...     'crossover1_freq': 2000.0     # Mid-High split
#             ... })

#         Neural Network Control:
#             >>> # Create parameter prediction network
#             >>> class MultiBandNoiseGateNet(nn.Module):
#             ...     def __init__(self, input_size, num_bands):
#             ...         super().__init__()
#             ...         num_params = num_bands * 5 + (num_bands - 1)
#             ...         self.net = nn.Sequential(
#             ...             nn.Linear(input_size, 32),
#             ...             nn.ReLU(),
#             ...             nn.Linear(32, num_params),
#             ...             nn.Sigmoid()  # Ensures output is in [0,1] range
#             ...         )
#             ...     
#             ...     def forward(self, x):
#             ...         return self.net(x)
#             >>> 
#             >>> # Process with predicted parameters
#             >>> controller = MultiBandNoiseGateNet(input_size=16, num_bands=3)
#             >>> norm_params = controller(features)  # Shape: [batch_size, 17]
#             >>> output = mb_gate(input_audio, norm_params=norm_params)
#     """
#     def __init__(self, sample_rate=44100, param_range=None, num_bands=3, knee_type="hard", smooth_type="ballistics"):
#         """Initialize the multi-band noise gate.

#         Args:
#             sample_rate (int): Audio sample rate in Hz
#             num_bands (int, optional): Number of frequency bands. Defaults to 3.
#             knee_type (str, optional): Type of gating knee curve. 
#                 Must be one of: "hard", "quadratic", "exponential". Defaults to "hard".
#             smooth_type (str, optional): Type of envelope follower.
#                 Must be one of: "ballistics", "iir". Defaults to "ballistics".
        
#         Raises:
#             ValueError: If knee_type is not one of "hard", "quadratic", or "exponential"
#             ValueError: If smooth_type is not one of "ballistics" or "iir"
#         """
#         self.num_bands = num_bands
#         super().__init__(sample_rate, param_range, num_bands, knee_type, smooth_type)
        
#     def _register_default_parameters(self):
#         """Register default parameter ranges for the multi-band noise gate.
    
#         Sets up the following parameters for each band i (0 to num_bands-1):
#             - band{i}_threshold_db: Threshold level (-90 to -20 dB)
#             - band{i}_ratio: Gating ratio (1 to 100)
#             - band{i}_knee_db: Knee width (0 to 6 dB)
#             - band{i}_attack_ms: Attack time (0.1 to 20 ms)
#             - band{i}_release_ms: Release time (5 to 1000 ms)
        
#         Also registers crossover frequencies:
#             - crossover{i}_freq: Split frequency between bands i and i+1
#                 with logarithmic spacing between 20 Hz and 20 kHz
        
#         Note:
#             Parameter ranges are stored in self.params dictionary and used for
#             mapping normalized parameters to DSP values.
#         """
#         self.params = {}
        
#         # Register parameters for each band
#         for i in range(self.num_bands):
#             band_prefix = f'band{i}_'
#             self.params.update({
#                 f'{band_prefix}threshold_db': EffectParam(min_val=-90.0, max_val=-20.0),
#                 f'{band_prefix}ratio': EffectParam(min_val=1.0, max_val=100.0),  # High ratios for aggressive gating
#                 f'{band_prefix}knee_db': EffectParam(min_val=0.0, max_val=6.0),
#                 f'{band_prefix}attack_ms': EffectParam(min_val=0.1, max_val=20.0),
#                 f'{band_prefix}release_ms': EffectParam(min_val=5.0, max_val=1000.0)
#             })
        
#         # Crossover frequencies between bands
#         for i in range(self.num_bands - 1):
#             min_freq = 20.0 * (2 ** i)  # Logarithmic spacing
#             max_freq = min(20000.0, min_freq * 100)
#             self.params[f'crossover{i}_freq'] = EffectParam(min_val=min_freq, max_val=max_freq)
    
#     def _compute_gain(self, level_db: torch.Tensor, threshold_db: torch.Tensor,
#                      ratio: torch.Tensor, knee_db: torch.Tensor) -> torch.Tensor:
#         """Compute noise gate gain for each band.
        
#         A noise gate is an extreme expander that applies aggressive attenuation
#         below the threshold. The gain computation follows the same mathematical
#         form as expansion but with higher ratios.
        
#         Args:
#             level_db (torch.Tensor): Input level in dB. Shape: (batch, time)
#             threshold_db (torch.Tensor): Threshold in dB. Shape: (batch,)
#             ratio (torch.Tensor): Gating ratio. Shape: (batch,)
#             knee_db (torch.Tensor): Knee width in dB. Shape: (batch,)
                
#         Returns:
#             torch.Tensor: Gain reduction in dB. Shape: (batch, time)
            
#         Note:
#             Implements three knee types:
#                 - "hard": Sharp transition at threshold
#                 - "quadratic": Smooth quadratic transition around threshold
#                 - "exponential": Continuous transition using softplus
            
#             All input tensors are automatically broadcast to match dimensions.
#         """
#         threshold_db = threshold_db.unsqueeze(-1)  # Shape: (batch, 1)
#         ratio = ratio.unsqueeze(-1)  # Shape: (batch, 1)
#         knee_db = knee_db.unsqueeze(-1)  # Shape: (batch, 1)
        
#         if self.knee_type == "hard":
#             # Simple threshold-based gating
#             below_thresh = level_db < threshold_db
#             gain_db = torch.where(
#                 below_thresh,
#                 (1 - 1 / ratio) * (level_db - threshold_db),
#                 torch.zeros_like(level_db)
#             )
            
#         elif self.knee_type == "quadratic":
#             knee_width = knee_db
#             below_knee = level_db < (threshold_db - knee_width / 2)
#             above_knee = level_db > (threshold_db + knee_width / 2)
            
#             # Below knee
#             gain_below = (1 - 1 / ratio) * (level_db - threshold_db)
#             # Above knee
#             gain_above = torch.zeros_like(level_db)
#             # In knee (centered at threshold)
#             x = (level_db - threshold_db + knee_width / 2) / knee_width
#             x = torch.clamp(x, 0.0, 1.0)
#             gain_knee = (1 - 1 / ratio) * (level_db - threshold_db + knee_width / 2) ** 2 / (2 * knee_width)
            
#             gain_db = (
#                 below_knee * gain_below +
#                 above_knee * gain_above +
#                 (~below_knee & ~above_knee) * gain_knee
#             )
            
#         else:  # "exponential"
#             knee_factor = torch.exp(knee_db)
#             x = level_db - threshold_db
#             gain_db = (1 - 1 / ratio) * torch.log1p(torch.exp(knee_factor * x)) / knee_factor
#             gain_db = torch.where(level_db < threshold_db, -gain_db.abs(), torch.zeros_like(level_db))

#         return gain_db
    
#     def _process_band(self, x: torch.Tensor, band_params: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """Process a single frequency band with noise gating.
    
#         Args:
#             x (torch.Tensor): Input audio for this band. 
#                 Shape: (batch, channels, samples)
#             band_params (Dict[str, torch.Tensor]): Gating parameters for this band.
#                 Must contain:
#                     - threshold_db: Threshold level in dB
#                     - ratio: Gating ratio
#                     - knee_db: Knee width in dB
#                     - attack_ms: Attack time in ms
#                     - release_ms: Release time in ms
        
#         Returns:
#             torch.Tensor: Processed audio for this band.
#                 Shape: (batch, channels, samples)
                
#         Note:
#             Processing steps:
#             1. Compute RMS level in dB
#             2. Apply envelope following using attack/release
#             3. Compute gain using knee characteristic
#             4. Apply gain
#         """
#         # Compute input energy and convert to dB
#         energy = x.square().mean(dim=-2)
#         level_db = 10 * torch.log10(energy + 1e-10)
        
#         # Convert time constants to z_alpha
#         if self.smoothing_type == "ballistics":
#             z_alpha = torch.stack([
#                 ms_to_z_alpha(band_params['attack_ms'], self.sample_rate),
#                 ms_to_z_alpha(band_params['release_ms'], self.sample_rate)
#             ], dim=-1)
#             smoothed_db = self.ballistics(level_db, z_alpha)
#         else:  # "iir"
#             avg_ms = (band_params['attack_ms'] + band_params['release_ms']) / 2
#             z_alpha = ms_to_z_alpha(avg_ms, self.sample_rate)
#             smoothed_db = self.iir_filter(level_db, z_alpha)
        
#         # Compute gain in dB
#         gain_db = self._compute_gain(
#             smoothed_db,
#             band_params['threshold_db'],
#             band_params['ratio'],
#             band_params['knee_db']
#         )
        
#         # Convert to linear gain and apply
#         gain_linear = torch.pow(10, gain_db / 20)
#         return gain_linear.unsqueeze(-2) * x
    
    
