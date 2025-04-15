import torch 
from typing import Dict, Union
from ..base import EffectParam
from ..core.envelope import Ballistics
from ..core.utils import ms_to_z_alpha
from .compressor import Compressor, MultiBandCompressor

# Limiter 
class Limiter(Compressor):
    """Differentiable feedforward peak limiter.
    
    A specialized compressor with high ratio and fast attack time to prevent audio signals from exceeding a specified threshold.

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        param_range (Dict[str, EffectParam]): Parameter ranges for the limiter.
    
    Parameters:
        threshold_db: Level at which limiting begins (-60 to 0 dB)
        ratio: Amount of gain reduction above threshold (20 to 100)
        knee_db: Width of the transition region (0 to 1 dB)
        attack_ms: Time taken to react to increases in level (0.1 to 1.0 ms)
        release_ms: Time taken to react to decreases in level (5 to 500 ms)
        makeup_db: Gain applied after limiting (-24 to 24 dB)

    Note:
        - Uses hard-knee characteristic for precise limiting
        - Employs extremely fast attack times (< 1ms)
        - Uses very high ratios (>20:1) for "brick wall" limiting
        - Has very narrow knee width for sharp transitions
        - Optimized for peak control rather than dynamic range control

    Warning:
        When using with neural networks:
        - norm_params must be in range [0, 1]
        - Parameters will be automatically mapped to their DSP ranges
        - Parameter ranges are more extreme than standard compression
        - Ensure network output dimension matches total parameters
        - Parameter order must match _register_default_parameters()
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
            - ratio: Limiting ratio (20 to 100)
            - knee_db: Knee width (0 to 1 dB)
            - attack_ms: Attack time (0.1 to 1.0 ms)
            - release_ms: Release time (5 to 500 ms)
            - makeup_db: Makeup gain (-12 to 12 dB)
            
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
    
    A specialized limiter that splits the input signal into multiple frequency bands and applies limiting independently to each band.

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        num_bands (int): Number of frequency bands. Defaults to 4.
        param_range (Dict[str, EffectParam]): Parameter ranges for the limiter.
        
    Parameters:
        crossover_freqs: Crossover frequencies between bands in Hz
            - Shape: (num_bands-1,)
            - Range: 20 to 20000 Hz
        threshold_db: Level at which limiting begins for each band
            - Shape: (num_bands,)
            - Range: -60 to 0 dB
        ratio: Amount of gain reduction above threshold for each band
            - Shape: (num_bands,)
            - Range: 20 to 100
        knee_db: Width of the transition region for each band
            - Shape: (num_bands,)
            - Range: 0 to 1 dB
        attack_ms: Time taken to react to increases in level for each band
            - Shape: (num_bands,)
            - Range: 0.1 to 1.0 ms
        release_ms: Time taken to react to decreases in level for each band
            - Shape: (num_bands,)
            - Range: 5 to 500 ms
        makeup_db: Gain applied after limiting for each band
            - Shape: (num_bands,)
            - Range: -12 to 12 dB

    Note:
        - Uses hard-knee characteristic for precise limiting
        - Employs extremely fast attack times (< 1ms)
        - Uses very high ratios (>20:1) for "brick wall" limiting
        - Has very narrow knee width for sharp transitions
        - Optimized for peak control rather than dynamic range control
        - Each band can be controlled independently

    Warning:
        When using with neural networks:
        - norm_params must be in range [0, 1]
        - Parameters will be automatically mapped to their DSP ranges
        - Parameter ranges are more extreme than standard compression
        - Ensure network output dimension matches total parameters
        - Parameter order must match _register_default_parameters()
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
            - ratio: Limiting ratio (20 to 100)
            - knee_db: Knee width (0 to 1 dB)
            - attack_ms: Attack time (0.1 to 1.0 ms)
            - release_ms: Release time (5 to 500 ms)
            - makeup_db: Makeup gain (-12 to 12 dB)
            
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
                f'{band_prefix}ratio': EffectParam(min_val=20.0, max_val=100.0),  # Higher ratio for limiting
                f'{band_prefix}knee_db': EffectParam(min_val=0.0, max_val=1.0),    # Very small knee for hard limiting
                f'{band_prefix}attack_ms': EffectParam(min_val=0.1, max_val=1.0),  # Fast attack for true limiting
                f'{band_prefix}release_ms': EffectParam(min_val=5.0, max_val=500.0),  # Controlled release
                f'{band_prefix}makeup_db': EffectParam(min_val=-12.0, max_val=12.0)
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

