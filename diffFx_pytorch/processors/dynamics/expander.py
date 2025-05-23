import torch 
from typing import Dict, Union
from ..base import EffectParam
from ..base_utils import check_params
from ..core.envelope import TruncatedOnePoleIIRFilter, Ballistics
from ..core.utils import ms_to_z_alpha
from .compressor import Compressor, MultiBandCompressor

# Need to be check 

# Expander 
class Expander(Compressor):
    """Differentiable expander based on compressor implementation.
    
    An expander increases the dynamic range of the signal by reducing the level
    of signals that fall below the threshold. The amount of reduction is determined
    by the ratio parameter.
    """
    def _register_default_parameters(self):
        self.params = {
            'threshold_db': EffectParam(min_val=-80.0, max_val=0.0),
            'ratio': EffectParam(min_val=1.0, max_val=8.0),  
            'knee_db': EffectParam(min_val=0.0, max_val=6.0),
            'attack_ms': EffectParam(min_val=0.05, max_val=300.0),
            'release_ms': EffectParam(min_val=5.0, max_val=4000.0),
            'makeup_db': EffectParam(min_val=-12.0, max_val=12.0)
        }
        
        # Initialize the original filter implementations
        self.iir_filter = TruncatedOnePoleIIRFilter(iir_len=16384)
        self.ballistics = Ballistics()
        
        # Configuration
        self.knee_type = "quadratic"
        self.smoothing_type = "ballistics" # "iir" # "ballistics"
    
    def _compute_gain(self, 
        level_db: torch.Tensor, 
        threshold_db: torch.Tensor,
        ratio: torch.Tensor, 
        knee_db: torch.Tensor
    ) -> torch.Tensor:
        threshold_db = threshold_db.unsqueeze(-1)
        ratio = ratio.unsqueeze(-1)
        knee_db = knee_db.unsqueeze(-1)
        
        knee_width = knee_db / 2
        below_knee = level_db < (threshold_db - knee_width)
        above_knee = level_db > (threshold_db + knee_width)
        
        # Below knee - full expansion
        gain_below = (level_db - threshold_db) * (1/ratio - 1)
        
        # Above knee - no expansion
        gain_above = torch.zeros_like(level_db)
        
        # In knee - quadratic transition
        gain_knee = (1/ratio - 1) * (level_db - threshold_db + knee_width).pow(2) / (4 * knee_width)
        
        # Combine regions
        gain_db = (below_knee * gain_below + 
                above_knee * gain_above + 
                (~below_knee & ~above_knee) * gain_knee)
        
        return gain_db

    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
            
        # Compute input energy and convert to dB
        energy = x.square().mean(dim=-2)
        level_db = 10 * torch.log10(energy + 1e-10)
        
        # Convert time constants to z_alpha
        if self.smoothing_type == "ballistics":
            z_alpha = torch.stack([
                ms_to_z_alpha(params['attack_ms'], self.sample_rate),
                ms_to_z_alpha(params['release_ms'], self.sample_rate),
            ], dim=-1)
            smoothed_db = self.ballistics(level_db, z_alpha)
        else:  # "iir"
            avg_ms = (params['attack_ms'] + params['release_ms']) / 2
            z_alpha = ms_to_z_alpha(avg_ms, self.sample_rate)
            smoothed_db = self.iir_filter(level_db, z_alpha)
            
        # Compute gain in dB
        gain_db = self._compute_gain(
            smoothed_db,
            params['threshold_db'],
            params['ratio'],
            params['knee_db']
        )
        
        # Apply makeup gain if available
        if 'makeup_db' in params:
            gain_db = gain_db + params['makeup_db'].unsqueeze(-1)
            
        # Convert to linear gain and apply
        gain_linear = torch.pow(10, gain_db / 20)
        return gain_linear.unsqueeze(-2) * x

# MultiBand Expander
class MultiBandExpander(MultiBandCompressor):
    """Differentiable multi-band dynamic range expander."""
    
    def __init__(self, sample_rate, param_range=None, num_bands=3):
        super().__init__(sample_rate, param_range, num_bands)
        self.smoothing_type = "ballistics"
        
    def _register_default_parameters(self):
        self.params = {}
        
        # Register parameters for each band
        for i in range(self.num_bands):
            band_prefix = f'band{i}_'
            self.params.update({
                f'{band_prefix}threshold_db': EffectParam(min_val=-80.0, max_val=-30.0),
                f'{band_prefix}ratio': EffectParam(min_val=1.0, max_val=20.0),  # Lower max ratio for expander
                f'{band_prefix}knee_db': EffectParam(min_val=0.0, max_val=6.0),
                f'{band_prefix}attack_ms': EffectParam(min_val=0.05, max_val=300.0),  # Faster min attack
                f'{band_prefix}release_ms': EffectParam(min_val=5.0, max_val=4000.0)
            })
        
        # Crossover frequencies between bands
        for i in range(self.num_bands - 1):
            min_freq = 20.0 * (2 ** i)  # Logarithmic spacing
            max_freq = min(20000.0, min_freq * 100)
            self.params[f'crossover{i}_freq'] = EffectParam(min_val=min_freq, max_val=max_freq)

    def _compute_gain(self, level_db: torch.Tensor, threshold_db: torch.Tensor,
                 ratio: torch.Tensor, knee_db: torch.Tensor) -> torch.Tensor:
        """Compute expansion gain based on knee type."""
        
        threshold_db = threshold_db.unsqueeze(-1)  # Shape: (batch, 1)
        ratio = ratio.unsqueeze(-1)  # Shape: (batch, 1)
        knee_db = knee_db.unsqueeze(-1)  # Shape: (batch, 1)
        
        knee_width = knee_db / 2
        below_knee = level_db < (threshold_db - knee_width)
        above_knee = level_db > (threshold_db + knee_width)
        
        # Below knee - full expansion
        gain_below = (level_db - threshold_db) * (1/ratio - 1)
        
        # Above knee - no expansion
        gain_above = torch.zeros_like(level_db)
        
        # In knee - quadratic transition
        gain_knee = (1/ratio - 1) * (level_db - threshold_db + knee_width).pow(2) / (4 * knee_width)
        
        # Combine regions
        gain_db = (below_knee * gain_below + 
                above_knee * gain_above + 
                (~below_knee & ~above_knee) * gain_knee)
        
        return gain_db
    
    def _process_band(self, x: torch.Tensor, band_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single frequency band with expansion."""
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
        
        # Compute gain in dB
        gain_db = self._compute_gain(
            smoothed_db,
            band_params['threshold_db'],
            band_params['ratio'],
            band_params['knee_db']
        )
        
        # Convert to linear gain and apply
        gain_linear = torch.pow(10, gain_db / 20)
        return gain_linear.unsqueeze(-2) * x
    
  
    
