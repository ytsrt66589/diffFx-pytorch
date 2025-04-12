import torch 
import torch.nn as nn
from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.envelope import Ballistics
from ..core.utils import ms_to_z_alpha
from .compressor import Compressor, MultiBandCompressor

# Limiter 
class Limiter(Compressor):
    """Differentiable feedforward peak limiter.
    
    This processor implements a peak limiter as a specialized compressor with a high
    ratio and fast attack time. It prevents the audio signal from exceeding a specified
    threshold by applying aggressive gain reduction when needed.

    Implementation is based on the following papers:
    
    ..  [1] Giannoulis, D., Massberg, M., & Reiss, J. D. (2012).
            "Digital dynamic range compressor design—A tutorial and analysis."
            Journal of the Audio Engineering Society, 60(6), 399-408.
    ..  [2] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    
    Processing Chain:
        1. Level Detection: Compute RMS energy and convert to dB
        2. Envelope Following: Smooth level using extremely fast attack/release
        3. Gain Computation: Apply hard-knee limiting curve
        4. Gain Application: Convert to linear gain and apply to signal

    The limiter implements a simplified gain computation using a hard knee:

    .. math::

        g(x) = \\begin{cases}
            0, & \\text{if } x \\leq T \\\\
            (T + \\frac{x - T}{R}) - x, & \\text{if } x > T
        \\end{cases}

    where:
        - x: input level in dB
        - T: threshold in dB
        - R: ratio (typically very high, >20:1)

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.

    Attributes:
        knee_type (str): Fixed to "hard" for limiting behavior
        smoothing_type (str): Fixed to "ballistics" for precise control
        ballistics (Ballistics): Envelope follower for attack/release

    Parameters Details:
        threshold_db: Level at which limiting begins
            - Controls the maximum output level
            - More negative values reduce peaks more aggressively
            - Range: -60 to 0 dB
        ratio: Amount of gain reduction above threshold
            - Very high ratios for "brick wall" limiting
            - Higher ratios prevent overshoots more effectively
            - Range: 20 to 1000 (much higher than standard compression)
        knee_db: Width of the transition region
            - Very narrow for sharp limiting
            - Range: 0 to 1 dB (much narrower than compression)
        attack_ms: Time taken to react to increases in level
            - Extremely fast to catch peaks
            - Range: 0.1 to 1.0 ms (much faster than compression)
        release_ms: Time taken to react to decreases in level
            - Controls how quickly gain reduction is released
            - Range: 5 to 500 ms
        makeup_db: Gain applied after limiting
            - Compensates for level reduction
            - Range: -24 to 24 dB

    Note:
        The limiter differs from a standard compressor in several ways:
            - Uses only hard-knee characteristic for precise limiting
            - Employs extremely fast attack times (< 1ms)
            - Uses very high ratios (>20:1) for "brick wall" limiting
            - Has a very narrow knee width for sharp transitions
            - Optimized for peak control rather than dynamic range control

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their DSP ranges
            - Parameter ranges are more extreme than standard compression
            - Ensure network output dimension matches total parameters
            - Parameter order must match _register_default_parameters()
        
    Examples:
        Basic DSP Usage:
            >>> # Create a limiter
            >>> limiter = Limiter(sample_rate=44100)
            >>> # Process audio with dsp parameters
            >>> output = limiter(input_audio, dsp_params={
            ...     'threshold_db': -3.0,
            ...     'ratio': 100.0,
            ...     'knee_db': 0.1,
            ...     'attack_ms': 0.1,
            ...     'release_ms': 50.0,
            ...     'makeup_db': 0.0
            ... })

        Neural Network Control:
            >>> # 1. Create parameter prediction network
            >>> class LimiterController(nn.Module):
            ...     def __init__(self, input_size):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, 6),  # 6 parameters
            ...             nn.Sigmoid()  # Ensures output is in [0,1]
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Initialize processor and network
            >>> limiter = Limiter(sample_rate=44100)
            >>> controller = LimiterController(input_size=16)
            >>> 
            >>> # Process with predicted parameters
            >>> features = torch.randn(batch_size, 16)
            >>> norm_params = controller(features)
            >>> output = limiter(input_audio, norm_params=norm_params)
    """
    def __init__(self, sample_rate=44100, param_range=None):
        
        super().__init__(sample_rate, param_range)
        
        # Limiter specific configuration
        self.knee_type = "hard"
        self.smoothing_type = "ballistics"
        # Initialize the original filter implementations
        self.ballistics = Ballistics() # for smoothing 
        
    def _register_default_parameters(self):
        """Register default parameter ranges for the limiter.
    
        Sets up the following parameters with their ranges:
            - threshold_db: Threshold level (-60 to 0 dB)
            - ratio: Limiting ratio (20 to 1000)
            - knee_db: Knee width (0 to 1 dB)
            - attack_ms: Attack time (0.1 to 1.0 ms)
            - release_ms: Release time (5 to 500 ms)
            - makeup_db: Makeup gain (-24 to 24 dB)
            
        Note:
            Parameter ranges are more extreme than standard compression
            to achieve limiting behavior.
        """
        self.params = {
            'threshold_db': EffectParam(min_val=-60.0, max_val=0.0),
            'ratio': EffectParam(min_val=20.0, max_val=100.0),
            'knee_db': EffectParam(min_val=0.0, max_val=1.0),
            'attack_ms': EffectParam(min_val=0.1, max_val=1.0),
            'release_ms': EffectParam(min_val=5.0, max_val=500.0),
            'makeup_db': EffectParam(min_val=-12.0, max_val=12.0)
        }
        
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """Process audio through the limiter.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, torch.Tensor], optional): Direct DSP parameters.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
            
        Note:
            Uses parent Compressor class processing with specialized
            parameter ranges for limiting behavior.
        """
        return super().process(x, norm_params, dsp_params)

# MultiBand Limiter
class MultiBandLimiter(MultiBandCompressor):
    """Differentiable multi-band peak limiter with frequency-dependent threshold control.
    
    This processor combines multi-band processing with peak limiting by splitting the
    input signal into multiple frequency bands using Linkwitz-Riley crossover filters
    and applying independent limiting to each band. This allows for precise peak
    control across different frequency ranges.

    Implementation is based on the following references:
    
    ..  [1] Giannoulis, D., Massberg, M., & Reiss, J. D. (2012).
            "Digital dynamic range compressor design—A tutorial and analysis."
            Journal of the Audio Engineering Society, 60(6), 399-408.
    
    Processing Chain:
        1. Band Splitting: Split input into frequency bands using Linkwitz-Riley filters
        2. Per-band Processing:
            a. Level Detection: Compute RMS energy and convert to dB
            b. Envelope Following: Smooth level using fast attack/release
            c. Gain Computation: Apply hard-knee limiting curve
            d. Gain Application: Convert to linear gain and apply to band
        3. Band Summation: Sum all processed bands to create final output

    The limiter implements hard-knee gain computation for each band:

    .. math::

        g(x) = \\begin{cases}
            0, & \\text{if } x \\leq T \\\\
            (T + \\frac{x - T}{R}) - x, & \\text{if } x > T
        \\end{cases}

    where:
        - x: input level in dB
        - T: threshold in dB
        - R: ratio (typically very high, >20:1)

    Args:
        sample_rate (int): Audio sample rate in Hz
        num_bands (int, optional): Number of frequency bands. Defaults to 3.

    Attributes:
        num_bands (int): Number of frequency bands
        knee_type (str): Fixed to "hard" for limiting behavior
        smoothing_type (str): Fixed to "ballistics" for precise control
        ballistics (Ballistics): Envelope follower for attack/release
        crossovers (nn.ModuleList): List of Linkwitz-Riley crossover filters

    Parameters Details:
        For each band i (0 to num_bands-1):
            band{i}_threshold_db: Level at which limiting begins
                - Controls maximum output level for this band
                - More negative values reduce peaks more aggressively
                - Range: -60 to 0 dB
            band{i}_ratio: Amount of gain reduction above threshold
                - Very high ratios for "brick wall" limiting
                - Higher ratios prevent overshoots more effectively
                - Range: 20 to 1000 (much higher than compression)
            band{i}_knee_db: Width of the transition region
                - Very narrow for sharp limiting
                - Range: 0 to 1 dB (much narrower than compression)
            band{i}_attack_ms: Time to react to increases in level
                - Extremely fast to catch peaks
                - Range: 0.1 to 1.0 ms (much faster than compression)
            band{i}_release_ms: Time to react to decreases in level
                - Controls how quickly limiting is reduced
                - Range: 5 to 500 ms
            band{i}_makeup_db: Gain applied after limiting
                - Compensates for level reduction
                - Range: -24 to 24 dB
        
        Crossover frequencies:
            crossover{i}_freq: Frequency splitting points between bands
                - Logarithmically spaced between 20 Hz and 20 kHz
                - crossover{i}_freq splits bands i and i+1
                - Range: Varies based on band position

    Note:
        - Each band uses hard-knee limiting with independent thresholds
        - Uses extremely fast attack times for precise peak control
        - Band splitting is done in series for proper phase relationships
        - Parameters can be controlled via DSP values or normalized inputs
        - Particularly useful for controlling peaks in specific frequency ranges
          (e.g., separately limiting low and high frequency content)

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their DSP ranges
            - Each band requires 6 parameters plus crossover frequencies
            - Total parameters = num_bands * 6 + (num_bands - 1)
            - Parameter order must match _register_default_parameters()
            - Ensure network output dimension matches total parameters
        
    Examples:
        Basic DSP Usage:
            >>> # Create a 3-band limiter
            >>> mb_limiter = MultiBandLimiter(
            ...     sample_rate=44100,
            ...     num_bands=3
            ... )
            >>> # Process with direct DSP parameters
            >>> output = mb_limiter(input_audio, dsp_params={
            ...     'band0_threshold_db': -6.0,   # Low band
            ...     'band0_ratio': 100.0,
            ...     'band0_knee_db': 0.1,
            ...     'band0_attack_ms': 0.1,
            ...     'band0_release_ms': 50.0,
            ...     'band0_makeup_db': 3.0,
            ...     'band1_threshold_db': -3.0,   # Mid band
            ...     'band1_ratio': 100.0,
            ...     'band1_knee_db': 0.1,
            ...     'band1_attack_ms': 0.1,
            ...     'band1_release_ms': 30.0,
            ...     'band1_makeup_db': 2.0,
            ...     'band2_threshold_db': -2.0,   # High band
            ...     'band2_ratio': 100.0,
            ...     'band2_knee_db': 0.1,
            ...     'band2_attack_ms': 0.1,
            ...     'band2_release_ms': 20.0,
            ...     'band2_makeup_db': 1.0,
            ...     'crossover0_freq': 200.0,     # Low-Mid split
            ...     'crossover1_freq': 2000.0     # Mid-High split
            ... })

        Neural Network Control:
            >>> # 1. Create parameter prediction network
            >>> class MultiBandLimiterNet(nn.Module):
            ...     def __init__(self, input_size, num_bands):
            ...         super().__init__()
            ...         # 6 params per band + (num_bands-1) crossovers
            ...         num_params = num_bands * 6 + (num_bands - 1)
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 64),
            ...             nn.ReLU(),
            ...             nn.Linear(64, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, num_params),
            ...             nn.Sigmoid()  # Ensures output is in [0,1]
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Initialize processor and network
            >>> mb_limiter = MultiBandLimiter(
            ...     sample_rate=44100,
            ...     num_bands=3
            ... )
            >>> num_params = mb_limiter.count_num_parameters()  # 21 parameters
            >>> controller = MultiBandLimiterNet(
            ...     input_size=16,
            ...     num_bands=3
            ... )
            >>> 
            >>> # Process with predicted parameters
            >>> features = torch.randn(batch_size, 16)
            >>> norm_params = controller(features)
            >>> output = mb_limiter(input_audio, norm_params=norm_params)
    """
    def __init__(self, sample_rate, param_range=None, num_bands=3):
        """Initialize the multi-band limiter.

        Args:
            sample_rate (int): Audio sample rate in Hz
            num_bands (int, optional): Number of frequency bands. Defaults to 3.
        
        Note:
            Configures the processor with fixed hard-knee and ballistics smoothing
            for optimal limiting behavior across all bands.
        """
        super().__init__(sample_rate, param_range, num_bands)
        # Limiter specific configuration
        self.knee_type = "hard"
        self.smoothing_type = "ballistics"
        
    def _register_default_parameters(self):
        """Register default parameter ranges for the multi-band limiter.
    
        Sets up parameters for each band with ranges optimized for limiting:
            - threshold_db: Threshold level (-60 to 0 dB)
            - ratio: Limiting ratio (20 to 1000)
            - knee_db: Knee width (0 to 1 dB)
            - attack_ms: Attack time (0.1 to 1.0 ms)
            - release_ms: Release time (5 to 500 ms)
            - makeup_db: Makeup gain (-24 to 24 dB)
            
        Also registers crossover frequencies with logarithmic spacing.
        
        Note:
            Parameter ranges are more extreme than compression
            to achieve true limiting behavior in each band.
        """
        self.params = {}
        
        # Register parameters for each band
        for i in range(self.num_bands):
            band_prefix = f'band{i}_'
            self.params.update({
                f'{band_prefix}threshold_db': EffectParam(min_val=-60.0, max_val=0.0),
                f'{band_prefix}ratio': EffectParam(min_val=20.0, max_val=1000.0),  # Higher ratio for limiting
                f'{band_prefix}knee_db': EffectParam(min_val=0.0, max_val=1.0),    # Very small knee for hard limiting
                f'{band_prefix}attack_ms': EffectParam(min_val=0.1, max_val=1.0),  # Fast attack for true limiting
                f'{band_prefix}release_ms': EffectParam(min_val=5.0, max_val=500.0),  # Controlled release
                f'{band_prefix}makeup_db': EffectParam(min_val=-24.0, max_val=24.0)
            })
        
        # Crossover frequencies between bands
        for i in range(self.num_bands - 1):
            min_freq = 20.0 * (2 ** i)  # Logarithmic spacing
            max_freq = min(20000.0, min_freq * 100)
            self.params[f'crossover{i}_freq'] = EffectParam(min_val=min_freq, max_val=max_freq)
    
    def _process_band(self, x: torch.Tensor, band_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single frequency band with limiting.
        
        Args:
            x (torch.Tensor): Input audio for this band. 
                Shape: (batch, channels, samples)
            band_params (Dict[str, torch.Tensor]): Limiting parameters for this band.
                Must contain:
                    - threshold_db: Threshold level in dB
                    - ratio: Limiting ratio
                    - knee_db: Knee width in dB
                    - attack_ms: Attack time in ms
                    - release_ms: Release time in ms
                    - makeup_db: Makeup gain in dB
        
        Returns:
            torch.Tensor: Processed audio for this band.
                Shape: (batch, channels, samples)
                
        Note:
            Uses hard-knee limiting with very fast attack times
            for precise peak control in each frequency band.
        """
        # Compute input energy and convert to dB
        energy = x.square().mean(dim=-2)
        level_db = 10 * torch.log10(energy + 1e-10)
        
        # Convert time constants to z_alpha
        if self.smoothing_type == "ballistics":
            z_alpha = torch.stack([
                ms_to_z_alpha(band_params['attack_ms'], self.sample_rate),
                ms_to_z_alpha(band_params['release_ms'], self.sample_rate)
            ], dim=-1)
            smoothed_db = self.ballistics(level_db, z_alpha)
        else:  # "iir"
            avg_ms = (band_params['attack_ms'] + band_params['release_ms']) / 2
            z_alpha = ms_to_z_alpha(avg_ms, self.sample_rate)
            smoothed_db = self.iir_filter(level_db, z_alpha)
        
        # Compute gain in dB with hard-knee limiting
        gain_db = self._compute_gain(
            smoothed_db,
            band_params['threshold_db'],
            band_params['ratio'],
            band_params['knee_db']
        )
        
        # Apply makeup gain
        gain_db = gain_db + band_params['makeup_db'].unsqueeze(-1)
        
        # Convert to linear gain and apply
        gain_linear = torch.pow(10, gain_db / 20)
        return gain_linear.unsqueeze(-2) * x
    
    def _compute_gain(self, level_db: torch.Tensor, threshold_db: torch.Tensor,
                     ratio: torch.Tensor, knee_db: torch.Tensor) -> torch.Tensor:
        """Compute limiting gain using hard-knee characteristic.
        
        Args:
            level_db (torch.Tensor): Input level in dB. Shape: (batch, time)
            threshold_db (torch.Tensor): Threshold in dB. Shape: (batch,)
            ratio (torch.Tensor): Limiting ratio. Shape: (batch,)
            knee_db (torch.Tensor): Knee width in dB. Shape: (batch,)
                
        Returns:
            torch.Tensor: Gain reduction in dB. Shape: (batch, time)
            
        Note:
            Implements hard-knee limiting with high ratio for
            "brick wall" style gain reduction.
        """
        threshold_db = threshold_db.unsqueeze(-1)  # Shape: (batch, 1)
        ratio = ratio.unsqueeze(-1)  # Shape: (batch, 1)
        
        # Hard knee limiting
        above_thresh = level_db > threshold_db
        gain_db = torch.where(
            above_thresh,
            (threshold_db + (level_db - threshold_db) / ratio) - level_db,
            torch.zeros_like(level_db)
        )
        
        return gain_db
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """Process audio through the multi-band limiter.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, torch.Tensor], optional): Direct DSP parameters.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input
            
        Note:
            Uses parent MultiBandCompressor processing chain with
            parameters optimized for limiting behavior in each band.
        """
        return super().process(x, norm_params, dsp_params)

