import torch 
from typing import Dict, Union
from ..base import EffectParam
from ..core.envelope import Ballistics
from .compressor import Compressor, MultiBandCompressor

# Limiter 
class Limiter(Compressor):
    """Differentiable feedforward peak limiter.
    A specialized compressor with high ratio and fast attack time to prevent audio signals from exceeding a specified threshold.
    """
    def __init__(self, sample_rate=44100, param_range=None):
        super().__init__(sample_rate, param_range)
        self.knee_type = "hard"
        self.smooth_filter = Ballistics() 
        
    def _register_default_parameters(self):
        
        self.params = {
            'threshold_db': EffectParam(min_val=-60.0, max_val=0.0),
            'ratio': EffectParam(min_val=20.0, max_val=100.0),
            'knee_db': EffectParam(min_val=0.0, max_val=1.0),
            'attack_ms': EffectParam(min_val=0.1, max_val=1.0),
            'release_ms': EffectParam(min_val=50.0, max_val=320.0),
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
        """
        return super().process(x, norm_params, dsp_params)

# MultiBand Limiter
class MultiBandLimiter(MultiBandCompressor):
    """Differentiable multi-band peak limiter with frequency-dependent threshold control. 
    A specialized limiter that splits the input signal into multiple frequency bands and applies limiting independently to each band.
    """
    def __init__(self, sample_rate, param_range=None, num_bands=3):
        """Initialize the multi-band limiter."""
        super().__init__(sample_rate, param_range, num_bands)
        self.knee_type = "hard"
        self.smooth_filter = Ballistics()
        
    def _register_default_parameters(self):
        """Register default parameter ranges for the multi-band limiter."""
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
    
    def _compute_gain(self, level_db: torch.Tensor, threshold_db: torch.Tensor,
                     ratio: torch.Tensor, knee_db: torch.Tensor) -> torch.Tensor:
        """Compute limiting gain using hard-knee characteristic."""
        threshold_db = threshold_db.unsqueeze(-1)  
        ratio = ratio.unsqueeze(-1)  
        
        above_thresh = level_db > threshold_db
        gain_db = torch.where(
            above_thresh,
            (threshold_db + (level_db - threshold_db) / ratio) - level_db,
            torch.zeros_like(level_db)
        )
        
        return gain_db
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """Process audio through the multi-band limiter."""
        return super().process(x, norm_params, dsp_params)

