import torch 
from typing import Dict, Union
from ..base import EffectParam
from ..base_utils import check_params
from ..core.envelope import Ballistics
from ..core.utils import ms_to_alpha
from .compressor import Compressor

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
            # 'makeup_db': EffectParam(min_val=-12.0, max_val=12.0)
        }
        
        self.smooth_filter = Ballistics() # for smoothing 
    
    def _compute_gain(self, 
        level_db: torch.Tensor, 
        threshold_db: torch.Tensor,
        ratio: torch.Tensor, 
        knee_db: torch.Tensor
    ) -> torch.Tensor:
        """Compute expansion gain.
        
        An expander reduces the level of signals below the threshold,
        increasing dynamic range.
        """
        threshold_db = threshold_db.unsqueeze(-1)
        ratio = ratio.unsqueeze(-1)
        knee_db = knee_db.unsqueeze(-1)
        
        knee_width = knee_db
        knee_start = threshold_db - knee_width / 2
        knee_end = threshold_db + knee_width / 2
        
        # Define regions
        below_knee = level_db < knee_start
        above_knee = level_db > knee_end
        in_knee = (~below_knee) & (~above_knee)
        
        # Below knee - full expansion (attenuation)
        gain_below = (1 - 1 / ratio) * (level_db - threshold_db)
        
        # Above knee - no expansion (unity gain = 0 dB)
        gain_above = torch.zeros_like(level_db)
        
        # In knee - smooth quadratic transition
        # Normalize position in knee (0 to 1)
        knee_pos = (level_db - knee_start) / knee_width  
        knee_pos = torch.clamp(knee_pos, 0.0, 1.0)
        
        # Quadratic interpolation from full expansion to no expansion
        # At knee_pos=0: full expansion, at knee_pos=1: no expansion
        expansion_amount = (1 - 1 / ratio) * (-knee_width / 2)  # Max expansion at knee start
        gain_knee = expansion_amount * (1 - knee_pos) ** 2
        
        # Combine regions
        gain_db = (
            below_knee.float() * gain_below +
            above_knee.float() * gain_above +
            in_knee.float() * gain_knee
        )
        
        return gain_db

    def _compute_level_db(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS level in dB for better musical behavior."""
        # RMS calculation with small window for responsiveness
        eps = 1e-8
        # Use RMS instead of peak detection
        x_squared = x ** 2
        # Simple moving average approximation (can be replaced with proper RMS)
        rms = torch.sqrt(x_squared.clamp(eps))
        return 20 * torch.log10(rms.clamp(eps))
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
            
        # Reshape parameters for broadcasting
        threshold_db = params['threshold_db'].view(-1, 1, 1)
        ratio = params['ratio'].view(-1, 1, 1)
        attack_ms = params['attack_ms'].view(-1, 1, 1)
        release_ms = params['release_ms'].view(-1, 1, 1)
        knee_db = params['knee_db'].view(-1, 1, 1)
        # makeup_db = params['makeup_db'].view(-1, 1, 1)

        bs, chs, seq_len = x.size()
        
        
        # Create side-chain from sum of channels
        x_side = x.mean(dim=1, keepdim=True)
        x_side = x_side.view(-1, 1, seq_len)
        eff_bs = x_side.size(0)
        # Compute input energy and convert to dB
        #eps = 1e-8
        #x_db = 20 * torch.log10(torch.abs(x_side).clamp(eps))
        x_db = self._compute_level_db(x_side)

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
        g_s = g_c_smooth # + makeup_db
        
        # Convert dB gains back to linear
        g_lin = 10 ** (g_s / 20.0)
        
        # Apply time-varying gain
        y = x * g_lin
        
        # Move channels back to the channel dimension
        y = y.view(bs, chs, seq_len)
        return y

# # MultiBand Expander
# class MultiBandExpander(MultiBandCompressor):
#     """Differentiable multi-band dynamic range expander.
    
#     This processor splits the input signal into multiple frequency bands using
#     Linkwitz-Riley crossover filters and applies independent expansion to each band.
#     The processed bands are then summed to produce the final output.
    
#     An expander increases the dynamic range of the signal by reducing the level
#     of signals that fall below the threshold. The amount of reduction is determined
#     by the ratio parameter.
    
#     Processing Chain:
#         1. Band Splitting: Split input into frequency bands using Linkwitz-Riley filters
#         2. Per-band Processing:
#             a. Level Detection: Compute RMS energy and convert to dB
#             b. Envelope Following: Smooth level using attack/release ballistics
#             c. Gain Computation: Apply expansion curve based on knee type
#             d. Gain Application: Convert to linear gain and apply to band
#         3. Band Summation: Sum all processed bands to create final output

#     Args:
#         sample_rate (int): Audio sample rate in Hz
#         param_range (Dict[str, EffectParam], optional): Parameter ranges.
#         num_bands (int, optional): Number of frequency bands. Defaults to 3.
#         knee_type (str, optional): Type of expansion knee curve. 
#             Must be one of: "hard", "quadratic", "exponential". Defaults to "quadratic".
#         smooth_type (str, optional): Type of envelope follower.
#             Must be one of: "ballistics", "iir". Defaults to "ballistics".

#     Parameters Details:
#         For each band i (0 to num_bands-1):
#             band{i}_threshold_db: Level at which expansion begins
#                 - Controls where expansion starts for this band (-80 to -30 dB)
#             band{i}_ratio: Amount of gain reduction below threshold
#                 - Higher ratios mean more expansion (1 to 8)
#             band{i}_knee_db: Width of transition region around threshold
#                 - Controls how gradually expansion is applied (0 to 6 dB)
#             band{i}_attack_ms: Time to react to level increases
#                 - Controls response to transients (0.05 to 300 ms)
#             band{i}_release_ms: Time to react to level decreases
#                 - Controls recovery time (5 to 4000 ms)
        
#         Crossover frequencies:
#             crossover{i}_freq: Split frequency between bands i and i+1
#                 - Logarithmically spaced between bands
#                 - Range varies based on position (20 Hz to 20 kHz)

#     Note:
#         - Uses 4th-order Linkwitz-Riley crossover filters
#         - Band splitting is done in series for proper phase alignment
#         - Each band has independent expansion parameters
#         - Parameters can be controlled via normalized (0-1) or DSP values
#         - Total parameters = num_bands * 5 + (num_bands - 1)

#     Examples:
#         Basic Usage:
#             >>> # Create a 3-band expander
#             >>> mb_exp = MultiBandExpander(
#             ...     sample_rate=44100,
#             ...     num_bands=3
#             ... )
#             >>> # Process with DSP parameters
#             >>> output = mb_exp(input_audio, dsp_params={
#             ...     'band0_threshold_db': -60.0,  # Low band
#             ...     'band0_ratio': 3.0,
#             ...     'band0_knee_db': 3.0,
#             ...     'band0_attack_ms': 5.0,
#             ...     'band0_release_ms': 100.0,
#             ...     'band1_threshold_db': -50.0,  # Mid band
#             ...     'band1_ratio': 2.5,
#             ...     'band1_knee_db': 3.0,
#             ...     'band1_attack_ms': 3.0,
#             ...     'band1_release_ms': 50.0,
#             ...     'band2_threshold_db': -40.0,  # High band
#             ...     'band2_ratio': 2.0,
#             ...     'band2_knee_db': 3.0,
#             ...     'band2_attack_ms': 1.0,
#             ...     'band2_release_ms': 20.0,
#             ...     'crossover0_freq': 200.0,     # Low-Mid split
#             ...     'crossover1_freq': 2000.0     # Mid-High split
#             ... })
#     """
    
#     def __init__(self, sample_rate, param_range=None, num_bands=3, knee_type="quadratic", smooth_type="ballistics"):
#         """Initialize the multi-band expander.

#         Args:
#             sample_rate (int): Audio sample rate in Hz
#             num_bands (int, optional): Number of frequency bands. Defaults to 3.
#             knee_type (str, optional): Type of expansion knee curve. 
#                 Must be one of: "hard", "quadratic", "exponential". Defaults to "quadratic".
#             smooth_type (str, optional): Type of envelope follower.
#                 Must be one of: "ballistics", "iir". Defaults to "ballistics".
        
#         Raises:
#             ValueError: If knee_type is not one of "hard", "quadratic", or "exponential"
#             ValueError: If smooth_type is not one of "ballistics" or "iir"
#         """
#         self.num_bands = num_bands
#         super().__init__(sample_rate, param_range, num_bands, knee_type, smooth_type)
        
#     def _register_default_parameters(self):
#         """Register default parameter ranges for the multi-band expander.
    
#         Sets up the following parameters for each band i (0 to num_bands-1):
#             - band{i}_threshold_db: Threshold level (-80 to -30 dB)
#             - band{i}_ratio: Expansion ratio (1 to 8)
#             - band{i}_knee_db: Knee width (0 to 6 dB)
#             - band{i}_attack_ms: Attack time (0.05 to 300 ms)
#             - band{i}_release_ms: Release time (5 to 4000 ms)
        
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
#                 f'{band_prefix}threshold_db': EffectParam(min_val=-80.0, max_val=-30.0),
#                 f'{band_prefix}ratio': EffectParam(min_val=1.0, max_val=8.0),
#                 f'{band_prefix}knee_db': EffectParam(min_val=0.0, max_val=6.0),
#                 f'{band_prefix}attack_ms': EffectParam(min_val=0.05, max_val=300.0),
#                 f'{band_prefix}release_ms': EffectParam(min_val=5.0, max_val=4000.0)
#             })
        
#         # Crossover frequencies between bands
#         for i in range(self.num_bands - 1):
#             min_freq = 20.0 * (2 ** i)  # Logarithmic spacing
#             max_freq = min(20000.0, min_freq * 100)
#             self.params[f'crossover{i}_freq'] = EffectParam(min_val=min_freq, max_val=max_freq)

#     def _compute_gain(self, level_db: torch.Tensor, threshold_db: torch.Tensor,
#                  ratio: torch.Tensor, knee_db: torch.Tensor) -> torch.Tensor:
#         """Compute expansion gain based on knee type.
    
#         Implementation based on standard expander mathematics with different knee characteristics.
        
#         Args:
#             level_db (torch.Tensor): Input level in dB. Shape: (batch, time)
#             threshold_db (torch.Tensor): Threshold in dB. Shape: (batch,)
#             ratio (torch.Tensor): Expansion ratio. Shape: (batch,)
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
        
        
#         knee_width = knee_db / 2
#         below_knee = level_db < (threshold_db - knee_width)
#         above_knee = level_db > (threshold_db + knee_width)
        
#         # Below knee - full expansion
#         gain_below = (threshold_db + (level_db - threshold_db) / ratio) - level_db
        
#         # Above knee - no expansion
#         gain_above = torch.zeros_like(level_db)
        
#         # In knee - quadratic interpolation
#         gain_knee = (1 / ratio - 1) * (level_db - threshold_db + knee_width).pow(2) / (4 * knee_width)
        
#         # Combine all regions
#         gain_db = (below_knee * gain_below + 
#                 above_knee * gain_above + 
#                 (~below_knee & ~above_knee) * gain_knee)
            
#         return gain_db
    
#     def _process_band(self, x: torch.Tensor, band_params: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """Process a single frequency band with expansion.
    
#         Args:
#             x (torch.Tensor): Input audio for this band. 
#                 Shape: (batch, channels, samples)
#             band_params (Dict[str, torch.Tensor]): Expansion parameters for this band.
#                 Must contain:
#                     - threshold_db: Threshold level in dB
#                     - ratio: Expansion ratio
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
    
  
    
