import torch 
import torch.nn as nn
import math 
from typing import Dict, Union, List
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.envelope import Ballistics
from ..core.utils import ms_to_alpha
from ..filters import LinkwitzRileyFilter

class Compressor(ProcessorsBase):
    """Differentiable feedforward dynamic range compressor.
    
    This processor implements a feedforward compressor with configurable knee curves and envelope following methods.
    The compressor uses level detection, smoothing, and gain computation stages to reduce dynamic range.

    Implementation is based on the following papers: 
    
    ..  [1] Giannoulis, Dimitrios, Michael Massberg, and Joshua D. Reiss. 
            "Digital dynamic range compressor design—A tutorial and analysis." 
            Journal of the Audio Engineering Society 60.6 (2012): 399-408.
    ..  [2] Lee, Sungho, et al. "GRAFX: an open-source library for audio processing graphs in PyTorch." 
            arXiv preprint arXiv:2408.03204 (2024).
    ..  [3] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    ..  [4] Yu, Chin-Yun, et al. "Differentiable all-pole filters for time-varying audio systems." 
            arXiv preprint arXiv:2404.07970 (2024).
    
    Processing Chain:
        1. Level Detection: RMS energy → dB
        2. Envelope Following: Attack/release smoothing
        3. Gain Computation: Apply compression curve
        4. Gain Application: Convert to linear gain and apply

    The compressor implements the following gain computation:

    Hard Knee:
        .. math::

            g(x) = \\begin{cases}
                0, & \\text{if } x \\leq T \\\\
                (T + \\frac{x - T}{R}) - x, & \\text{if } x > T
            \\end{cases}

    Quadratic Knee:
        .. math::

            g(x) = \\begin{cases}
                0, & \\text{if } x \\leq T - W/2 \\\\
                (\\frac{1}{R} - 1)\\frac{(x - T + W/2)^2}{4W}, & \\text{if } T - W/2 < x \\leq T + W/2 \\\\
                (T + \\frac{x - T}{R}) - x, & \\text{if } x > T + W/2
            \\end{cases}

    Exponential Knee:
        .. math::

            g(x) = (\\frac{1}{R} - 1) \\frac{\\text{softplus}(k(x - T))}{k}

        where k = exp(W) is the knee factor

    Variables:
        - x: input level in dB
        - T: threshold in dB
        - R: ratio
        - W: knee width in dB

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        knee_type (str, optional): Type of compression knee curve. 
            Must be one of: "hard", "quadratic", "exponential". Defaults to "quadratic".
        smooth_type (str, optional): Type of envelope follower.
            Must be one of: "ballistics", "iir". Defaults to "ballistics".

    Attributes:
        knee_type (str): Current knee characteristic type
        smoothing_type (str): Current envelope follower type
        ballistics (Ballistics): Envelope follower for attack/release
        iir_filter (TruncatedOnePoleIIRFilter): IIR filter for smoothing

    Parameters Details:
        threshold_db: Level at which compression begins
            - Controls where compression starts to take effect
            - More negative values compress more of the signal
        ratio: Amount of gain reduction above threshold
            - 2:1 means for every 2 dB increase in input, output increases by 1 dB
            - Higher ratios mean more compression
            - 1:1 means no compression
        knee_db: Width of the transition region around threshold
            - 0 dB creates a hard knee (sudden transition)
            - Larger values create smoother transitions
            - Affects how gradually compression is applied
        attack_ms: Time taken to react to increases in level
            - Shorter times mean faster response to transients
            - Longer times let more of the transient through
        release_ms: Time taken to react to decreases in level
            - Controls how quickly compression is reduced
            - Affects the "movement" of the compression
        makeup_db: Gain applied after compression
            - Compensates for level reduction from compression

    Note:
        The processor supports the following parameter ranges (same as single-band Compressor):
            - threshold_db: Threshold level in dB (-60 to 0)
            - ratio: Compression ratio (1 to 20)
            - knee_db: Knee width in dB (0 to 12)
            - attack_ms: Attack time in ms (0.1 to 100)
            - release_ms: Release time in ms (10 to 1000)
            - makeup_db: Makeup gain in dB (-12 to 12)

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their DSP ranges
            - Ensure your network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()
        
    Examples
    --------
    Basic DSP Usage:
        >>> # Create a compressor with quadratic knee
        >>> compressor = Compressor(
        ...     sample_rate=44100,
        ...     knee_type="quadratic",
        ...     smooth_type="ballistics"
        ... )
        >>> # Process audio with dsp parameters
        >>> output = compressor(input_audio, dsp_params={
        ...     'threshold_db': -20.0,
        ...     'ratio': 4.0,
        ...     'knee_db': 6.0,
        ...     'attack_ms': 5.0,
        ...     'release_ms': 50.0,
        ...     'makeup_db': 0.0
        ... })

    Neural Network Control:
        >>> # 1. Simple parameter prediction
        >>> class CompressorController(nn.Module):
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
        >>> num_params = compressor.count_num_parameters()  # 6 parameters
        >>> controller = CompressorController(input_size=16, num_params=num_params)
        >>> 
        >>> # Process with features
        >>> features = torch.randn(batch_size, 16)  # Audio features
        >>> norm_params = controller(features)
        >>> output = compressor(input_audio, norm_params=norm_params)
    """
    def __init__(
        self, 
        sample_rate = 44100, 
        param_range = None, 
        knee_type = "hard"
    ):
        """Initialize the compressor.

        Args:
            sample_rate (int): Audio sample rate in Hz
            param_range (Dict[str, EffectParam], optional): Parameter ranges.
            knee_type (str): Type of compression knee curve
            smooth_type (str): Type of envelope follower
        
        Raises:
            ValueError: If knee_type or smooth_type is invalid
        """
        super().__init__(sample_rate, param_range)
        if knee_type not in ["hard", "quadratic", "exponential"]:
            raise ValueError("Invalid knee type, please choose hard, quadratic or exponential")
        
        self.knee_type = knee_type
        self.smooth_filter = Ballistics() 
        
    def _register_default_parameters(self):
        """Register default parameter ranges for the compressor.
    
        Sets up the following parameters with their ranges:
            - threshold_db: Threshold level (-60 to 0 dB)
            - ratio: Compression ratio (1 to 20)
            - knee_db: Knee width (0 to 12 dB)
            - attack_ms: Attack time (0.1 to 100 ms)
            - release_ms: Release time (10 to 1000 ms)
            - makeup_db: Makeup gain (-12 to 12 dB)
        """
        self.params = {
            'threshold_db': EffectParam(min_val=-60.0, max_val=0.0),
            'ratio': EffectParam(min_val=1.0, max_val=20.0),
            'knee_db': EffectParam(min_val=0.0, max_val=12.0),
            'attack_ms': EffectParam(min_val=0.1, max_val=100.0),
            'release_ms': EffectParam(min_val=10.0, max_val=1000.0),
            'makeup_db': EffectParam(min_val=-12.0, max_val=12.0) 
        }
             
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the compressor.

        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Dictionary with keys:
                - 'threshold_db': Level at which compression begins (-60 to 0 dB)
                - 'ratio': Amount of gain reduction above threshold (1 to 20)
                - 'knee_db': Width of transition region around threshold (0 to 12 dB)
                - 'attack_ms': Time to react to level increases (0.1 to 100 ms)
                - 'release_ms': Time to react to level decreases (10 to 1000 ms)
                - 'makeup_db': Gain applied after compression (-12 to 12 dB)
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
        # Process parameters
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        bs, chs, seq_len = x.size()
        
        # Create side-chain from sum of channels
        x_side = x.mean(dim=1, keepdim=True)
        x_side = x_side.view(-1, 1, seq_len)
        eff_bs = x_side.size(0)
        
        # Reshape parameters for broadcasting
        threshold_db = params['threshold_db'].view(-1, 1, 1)
        ratio = params['ratio'].view(-1, 1, 1)
        attack_ms = params['attack_ms'].view(-1, 1, 1)
        release_ms = params['release_ms'].view(-1, 1, 1)
        knee_db = params['knee_db'].view(-1, 1, 1)
        makeup_db = params['makeup_db'].view(-1, 1, 1)
        
        # Compute input energy and convert to dB
        eps = 1e-8
        x_db = 20 * torch.log10(torch.abs(x_side).clamp(eps))
        
        # Compute gain reduction using _compute_gain method
        g_c = self._compute_gain(
            x_db.squeeze(-2),  
            threshold_db.squeeze(-1),
            ratio.squeeze(-1),
            knee_db.squeeze(-1)
        )  
        

        # Convert time constants to z_alpha
        alpha = torch.stack([
            ms_to_alpha(attack_ms.squeeze(-1), self.sample_rate),
            ms_to_alpha(release_ms.squeeze(-1), self.sample_rate)
        ], dim=-1).squeeze(-2)
        g_c_smooth = self.smooth_filter(g_c.squeeze(-2), alpha).unsqueeze(-2)

        # Add makeup gain in dB
        g_s = g_c_smooth + makeup_db
        
        # Convert dB gains back to linear
        g_lin = 10 ** (g_s / 20.0)
        
        # Apply time-varying gain
        y = x * g_lin
        
        # Move channels back to the channel dimension
        y = y.view(bs, chs, seq_len)
        
        return y

    def _compute_gain(self, level_db: torch.Tensor, threshold_db: torch.Tensor,
                     ratio: torch.Tensor, knee_db: torch.Tensor) -> torch.Tensor:
        """Compute compression gain based on knee type.
    
        Implementation based on [1], [2] with different knee characteristics.

        Args:
            level_db (torch.Tensor): Input level in dB. Shape: (batch, time)
            threshold_db (torch.Tensor): Threshold in dB. Shape: (batch,)
            ratio (torch.Tensor): Compression ratio. Shape: (batch,)
            knee_db (torch.Tensor): Knee width in dB. Shape: (batch,)
            
        Returns:
            torch.Tensor: Gain reduction in dB. Shape: (batch, time)
            
        Note:
            The gain computation depends on the knee_type:
            - "hard": Sharp transition at threshold
            - "quadratic": Smooth quadratic transition around threshold
            - "exponential": Continuous transition using softplus
        """
        threshold_db = threshold_db.unsqueeze(-1)  
        ratio = ratio.unsqueeze(-1) 
        knee_db = knee_db.unsqueeze(-1)  
        
        if self.knee_type == "hard":
            above_thresh = level_db > threshold_db
            gain_db = torch.where(
                above_thresh,
                (threshold_db + (level_db - threshold_db) / ratio) - level_db,
                torch.zeros_like(level_db)
            )
            
        elif self.knee_type == "quadratic":
            knee_width = knee_db / 2
            below_knee = level_db < (threshold_db - knee_width)
            above_knee = level_db > (threshold_db + knee_width)
            
            # Below knee
            gain_below = torch.zeros_like(level_db)
            
            # Above knee
            gain_above = (threshold_db + (level_db - threshold_db) / ratio) - level_db
            
            # In knee
            gain_knee = (1 / ratio - 1) * (level_db - threshold_db + knee_width).pow(2) / (4 * knee_width)
            
            # Combine
            gain_db = (below_knee * gain_below + 
                      above_knee * gain_above + 
                      (~below_knee & ~above_knee) * gain_knee)
            
        else:  # exponential
            knee_factor = torch.exp(knee_db)
            gain_db = ((1 / ratio - 1) * 
                      torch.nn.functional.softplus(knee_factor * (level_db - threshold_db)) / 
                      knee_factor)

        return gain_db

# MultiBand Compressor 
# MultiBand Compressor 
class MultiBandCompressor(ProcessorsBase):
    """Differentiable multi-band dynamic range compressor.
    
    This processor splits the input signal into multiple frequency bands using
    Linkwitz-Riley crossover filters and applies independent compression to each band.
    The processed bands are then summed to produce the final output.

    Implementation is based on the following references:
    
    ..  [1] MATLAB Audio Toolbox. "Multiband Dynamic Range Compression."
            https://www.mathworks.com/help/audio/ug/multiband-dynamic-range-compression.html
    ..  [2] Giannoulis, D., Massberg, M., & Reiss, J. D. (2012).
            "Digital dynamic range compressor design—A tutorial and analysis."
            Journal of the Audio Engineering Society, 60(6), 399-408.
    
    Processing Chain:
        1. Band Splitting: Split input into frequency bands using Linkwitz-Riley filters
        2. Per-band Processing:
            a. Level Detection: Compute RMS energy and convert to dB
            b. Envelope Following: Smooth level using attack/release ballistics
            c. Gain Computation: Apply compression curve based on knee type
            d. Gain Application: Convert to linear gain and apply to band
        3. Band Summation: Sum all processed bands to create final output

    Args:
        sample_rate (int): Audio sample rate in Hz
        param_range (Dict[str, EffectParam], optional): Parameter ranges.
        num_bands (int, optional): Number of frequency bands. Defaults to 3.
        knee_type (str, optional): Type of compression knee curve. 
            Must be one of: "hard", "quadratic", "exponential". Defaults to "quadratic".
        smooth_type (str, optional): Type of envelope follower.
            Must be one of: "ballistics", "iir". Defaults to "ballistics".

    Parameters Details:
        For each band i (0 to num_bands-1):
            band{i}_threshold_db: Level at which compression begins
                - Controls where compression starts for this band (-60 to 0 dB)
            band{i}_ratio: Amount of gain reduction above threshold
                - Higher ratios mean more compression (1 to 20)
            band{i}_knee_db: Width of transition region around threshold
                - Controls how gradually compression is applied (0 to 12 dB)
            band{i}_attack_ms: Time to react to level increases
                - Controls response to transients (1 to 500 ms)
            band{i}_release_ms: Time to react to level decreases
                - Controls recovery time (10 to 2000 ms)
            band{i}_makeup_db: Gain applied after compression
                - Compensates for level reduction (-24 to 24 dB)
        
        Crossover frequencies:
            crossover{i}_freq: Split frequency between bands i and i+1
                - Logarithmically spaced between bands
                - Range varies based on position (20 Hz to 20 kHz)

    Note:
        - Uses 4th-order Linkwitz-Riley crossover filters
        - Band splitting is done in series for proper phase alignment
        - Each band has independent compression parameters
        - Parameters can be controlled via normalized (0-1) or DSP values
        - Total parameters = num_bands * 6 + (num_bands - 1)

    Examples:
        Basic Usage:
            >>> # Create a 3-band compressor
            >>> mb_comp = MultiBandCompressor(
            ...     sample_rate=44100,
            ...     num_bands=3
            ... )
            >>> # Process with DSP parameters
            >>> output = mb_comp(input_audio, dsp_params={
            ...     'band0_threshold_db': -24.0,  # Low band
            ...     'band0_ratio': 4.0,
            ...     'band0_knee_db': 6.0,
            ...     'band0_attack_ms': 10.0,
            ...     'band0_release_ms': 100.0,
            ...     'band0_makeup_db': 3.0,
            ...     'band1_threshold_db': -18.0,  # Mid band
            ...     'band1_ratio': 3.0,
            ...     'band1_knee_db': 6.0,
            ...     'band1_attack_ms': 5.0,
            ...     'band1_release_ms': 50.0,
            ...     'band1_makeup_db': 2.0,
            ...     'band2_threshold_db': -12.0,  # High band
            ...     'band2_ratio': 2.0,
            ...     'band2_knee_db': 6.0,
            ...     'band2_attack_ms': 1.0,
            ...     'band2_release_ms': 20.0,
            ...     'band2_makeup_db': 1.0,
            ...     'crossover0_freq': 200.0,     # Low-Mid split
            ...     'crossover1_freq': 2000.0     # Mid-High split
            ... })

        Neural Network Control:
            >>> # Create parameter prediction network
            >>> class MultiBandCompressorNet(nn.Module):
            ...     def __init__(self, input_size, num_bands):
            ...         super().__init__()
            ...         num_params = num_bands * 6 + (num_bands - 1)
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
            >>> # Process with predicted parameters
            >>> controller = MultiBandCompressorNet(input_size=16, num_bands=3)
            >>> norm_params = controller(features)  # Shape: [batch_size, 21]
            >>> output = mb_comp(input_audio, norm_params=norm_params)
    """
    def __init__(self, sample_rate, param_range=None, num_bands=3, knee_type="hard"):
        """Initialize the multi-band compressor.

        Args:
            sample_rate (int): Audio sample rate in Hz
            num_bands (int, optional): Number of frequency bands. Defaults to 3.
            knee_type (str, optional): Type of compression knee curve. 
                Must be one of: "hard", "quadratic", "exponential". Defaults to "quadratic".
            smooth_type (str, optional): Type of envelope follower.
                Must be one of: "ballistics", "iir". Defaults to "ballistics".
        
        Raises:
            ValueError: If knee_type is not one of "hard", "quadratic", or "exponential"
            ValueError: If smooth_type is not one of "ballistics" or "iir"
            
        Note:
            Initializes crossover filters as a ModuleList with (num_bands - 1) filters
            to split the input into num_bands frequency bands.
        """
        self.num_bands = num_bands
        super().__init__(sample_rate, param_range)
        
        if knee_type not in ["hard", "quadratic", "exponential"]:
            raise ValueError("Invalid knee type, please choose hard, quadratic or exponential")
        
        self.knee_type = knee_type
        self.smooth_filter = Ballistics() # for smoothing 
        
        # Create crossover filters
        self.crossovers = nn.ModuleList([
            LinkwitzRileyFilter(sample_rate) 
            for _ in range(num_bands - 1)
        ])
        
    def _register_default_parameters(self):
        """Register default parameter ranges for the multi-band compressor.
    
        Sets up the following parameters for each band i (0 to num_bands-1):
            - band{i}_threshold_db: Threshold level (-60 to 0 dB)
            - band{i}_ratio: Compression ratio (1 to 20)
            - band{i}_knee_db: Knee width (0 to 12 dB)
            - band{i}_attack_ms: Attack time (1 to 500 ms)
            - band{i}_release_ms: Release time (10 to 2000 ms)
            - band{i}_makeup_db: Makeup gain (-24 to 24 dB)
        
        Also registers crossover frequencies:
            - crossover{i}_freq: Split frequency between bands i and i+1
                with logarithmic spacing between 20 Hz and 20 kHz
        
        Note:
            Parameter ranges are stored in self.params dictionary and used for
            mapping normalized parameters to DSP values.
        """
        self.params = {}
        
        # Register parameters for each band
        for i in range(self.num_bands):
            band_prefix = f'band{i}_'
            self.params.update({
                f'{band_prefix}threshold_db': EffectParam(min_val=-60.0, max_val=0.0),
                f'{band_prefix}ratio': EffectParam(min_val=1.0, max_val=20.0),
                f'{band_prefix}knee_db': EffectParam(min_val=0.0, max_val=12.0),
                f'{band_prefix}attack_ms': EffectParam(min_val=1.0, max_val=500.0),
                f'{band_prefix}release_ms': EffectParam(min_val=10.0, max_val=2000.0),
                f'{band_prefix}makeup_db': EffectParam(min_val=-12.0, max_val=12.0)
            })
        
        # Crossover frequencies between bands
        for i in range(self.num_bands - 1):
            min_freq = 20.0 * (2 ** i)  # Logarithmic spacing
            max_freq = min(20000.0, min_freq * 100)
            self.params[f'crossover{i}_freq'] = EffectParam(min_val=min_freq, max_val=max_freq)
    
    def _compute_gain(self, level_db: torch.Tensor, threshold_db: torch.Tensor,
                 ratio: torch.Tensor, knee_db: torch.Tensor) -> torch.Tensor:
        """Compute compression gain based on knee type.
    
        Implementation based on [2] with different knee characteristics.
        
        Args:
            level_db (torch.Tensor): Input level in dB. Shape: (batch, time)
            threshold_db (torch.Tensor): Threshold in dB. Shape: (batch,)
            ratio (torch.Tensor): Compression ratio. Shape: (batch,)
            knee_db (torch.Tensor): Knee width in dB. Shape: (batch,)
                
        Returns:
            torch.Tensor: Gain reduction in dB. Shape: (batch, time)
            
        Note:
            Implements three knee types:
                - "hard": Sharp transition at threshold
                - "quadratic": Smooth quadratic transition around threshold
                - "exponential": Continuous transition using softplus
            
            All input tensors are automatically broadcast to match dimensions.
        """
        threshold_db = threshold_db.unsqueeze(-1)  # Shape: (batch, 1)
        ratio = ratio.unsqueeze(-1)  # Shape: (batch, 1)
        knee_db = knee_db.unsqueeze(-1)  # Shape: (batch, 1)
        
        if self.knee_type == "hard":
            # Simple threshold-based compression
            above_thresh = level_db > threshold_db
            gain_db = torch.where(
                above_thresh,
                (threshold_db + (level_db - threshold_db) / ratio) - level_db,
                torch.zeros_like(level_db)
            )
            
        elif self.knee_type == "quadratic":
            knee_width = knee_db / 2
            below_knee = level_db < (threshold_db - knee_width)
            above_knee = level_db > (threshold_db + knee_width)
            
            # Below knee - no compression
            gain_below = torch.zeros_like(level_db)
            
            # Above knee - full compression
            gain_above = (threshold_db + (level_db - threshold_db) / ratio) - level_db
            
            # In knee - quadratic interpolation
            gain_knee = (1 / ratio - 1) * (level_db - threshold_db + knee_width).pow(2) / (4 * knee_width)
            
            # Combine all regions
            gain_db = (below_knee * gain_below + 
                    above_knee * gain_above + 
                    (~below_knee & ~above_knee) * gain_knee)
            
        else:  # "exponential"
            # Exponential knee using softplus for smooth transition
            knee_factor = torch.exp(knee_db)
            
            # Normalized input level relative to threshold
            x = level_db - threshold_db
            
            # Compute gain using softplus for smooth transition
            gain_db = ((1 / ratio - 1) * 
                    torch.nn.functional.softplus(knee_factor * x) / 
                    knee_factor)
            
            # Ensure no gain reduction below threshold
            gain_db = torch.min(gain_db, torch.zeros_like(gain_db))

        return gain_db
    
    def _process_band(self, x: torch.Tensor, band_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single frequency band with compression.
    
        Args:
            x (torch.Tensor): Input audio for this band. 
                Shape: (batch, channels, samples)
            band_params (Dict[str, torch.Tensor]): Compression parameters for this band.
                Must contain:
                    - threshold_db: Threshold level in dB
                    - ratio: Compression ratio
                    - knee_db: Knee width in dB
                    - attack_ms: Attack time in ms
                    - release_ms: Release time in ms
                    - makeup_db: Makeup gain in dB
        
        Returns:
            torch.Tensor: Processed audio for this band.
                Shape: (batch, channels, samples)
                
        Note:
            Processing steps:
            1. Compute RMS level in dB
            2. Apply envelope following using attack/release
            3. Compute gain using knee characteristic
            4. Apply gain and makeup
        """
        bs, chs, seq_len = x.size()
        
        # Create side-chain from sum of channels
        x_side = x.mean(dim=1, keepdim=True)
        x_side = x_side.view(-1, 1, seq_len)
        eff_bs = x_side.size(0)
        
        # Reshape parameters for broadcasting
        threshold_db = band_params['threshold_db'].view(-1, 1, 1)
        ratio = band_params['ratio'].view(-1, 1, 1)
        attack_ms = band_params['attack_ms'].view(-1, 1, 1)
        release_ms = band_params['release_ms'].view(-1, 1, 1)
        knee_db = band_params['knee_db'].view(-1, 1, 1)
        makeup_db = band_params['makeup_db'].view(-1, 1, 1)
        
        # Compute input energy and convert to dB
        eps = 1e-8
        x_db = 20 * torch.log10(torch.abs(x_side).clamp(eps))
        
        # Compute gain reduction using _compute_gain method
        g_c = self._compute_gain(
            x_db.squeeze(-2),  
            threshold_db.squeeze(-1),
            ratio.squeeze(-1),
            knee_db.squeeze(-1)
        )  
        

        # Convert time constants to z_alpha
        alpha = torch.stack([
            ms_to_alpha(attack_ms.squeeze(-1), self.sample_rate),
            ms_to_alpha(release_ms.squeeze(-1), self.sample_rate)
        ], dim=-1).squeeze(-2)
        g_c_smooth = self.smooth_filter(g_c.squeeze(-2), alpha).unsqueeze(-2)

        # Add makeup gain in dB
        g_s = g_c_smooth + makeup_db
        
        # Convert dB gains back to linear
        g_lin = 10 ** (g_s / 20.0)
        
        # Apply time-varying gain
        y = x * g_lin
        
        # Move channels back to the channel dimension
        y = y.view(bs, chs, seq_len)
        
        return y
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None,
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """Process input signal through the multi-band compressor.
    
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Dictionary with keys for each band i (0 to num_bands-1):
                - f'band{i}_threshold_db': Level at which compression begins (-60 to 0 dB)
                - f'band{i}_ratio': Amount of gain reduction above threshold (1 to 20)
                - f'band{i}_knee_db': Width of transition region around threshold (0 to 12 dB)
                - f'band{i}_attack_ms': Time to react to level increases (1 to 500 ms)
                - f'band{i}_release_ms': Time to react to level decreases (10 to 2000 ms)
                - f'band{i}_makeup_db': Gain applied after compression (-24 to 24 dB)
                And crossover frequencies between bands:
                - f'crossover{i}_freq': Split frequency between bands i and i+1
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
        check_params(norm_params, dsp_params)
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        # Split into frequency bands using LR crossovers
        bands = []
        current_signal = x
        
        # Apply crossovers in series to split bands
        for i, crossover in enumerate(self.crossovers):
            # Get separate low and high outputs from the crossover
            low, high = crossover.get_separate_outputs(
                current_signal, 
                norm_params=None, 
                dsp_params={'frequency': params[f'crossover{i}_freq']}
            )
            bands.append(low)
            current_signal = high
        
        # Add the final high band
        bands.append(current_signal)
        
        # Process each band with compression
        processed_bands = []
        for i, band in enumerate(bands):
            band_params = {
                key.replace(f'band{i}_', ''): value 
                for key, value in params.items() 
                if key.startswith(f'band{i}_')
            }
            processed_band = self._process_band(band, band_params)
            processed_bands.append(processed_band)
        
        # Sum all bands
        return sum(processed_bands)