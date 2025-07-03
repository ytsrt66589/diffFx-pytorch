import torch 
from typing import Dict, Union
from ..base import EffectParam
from ..base_utils import check_params
from ..core.envelope import Ballistics
from ..core.utils import ms_to_alpha
from .compressor import Compressor


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
        }
        
        self.smooth_filter = Ballistics() 
    
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
        
        below_knee = level_db < knee_start
        above_knee = level_db > knee_end
        in_knee = (~below_knee) & (~above_knee)
        
        gain_below = (1 - 1 / ratio) * (level_db - threshold_db)
        gain_above = torch.zeros_like(level_db)
        
        knee_pos = (level_db - knee_start) / knee_width  
        knee_pos = torch.clamp(knee_pos, 0.0, 1.0)
        
        expansion_amount = (1 - 1 / ratio) * (-knee_width / 2)  
        gain_knee = expansion_amount * (1 - knee_pos) ** 2
        
        gain_db = (
            below_knee.float() * gain_below +
            above_knee.float() * gain_above +
            in_knee.float() * gain_knee
        )
        
        return gain_db

    def _compute_level_db(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS level in dB for better musical behavior."""
        eps = 1e-8
        x_squared = x ** 2
        rms = torch.sqrt(x_squared.clamp(eps))
        return 20 * torch.log10(rms.clamp(eps))
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
            
        threshold_db = params['threshold_db'].view(-1, 1, 1)
        ratio = params['ratio'].view(-1, 1, 1)
        attack_ms = params['attack_ms'].view(-1, 1, 1)
        release_ms = params['release_ms'].view(-1, 1, 1)
        knee_db = params['knee_db'].view(-1, 1, 1)

        bs, chs, seq_len = x.size()
        
        
        # Create side-chain from sum of channels
        x_side = x.mean(dim=1, keepdim=True)
        x_side = x_side.view(-1, 1, seq_len)
        eff_bs = x_side.size(0)

        x_db = self._compute_level_db(x_side)

        g_c = self._compute_gain(
            x_db.squeeze(-2),  
            threshold_db.squeeze(-1),
            ratio.squeeze(-1),
            knee_db.squeeze(-1)
        )  
        
        alpha = torch.stack([
            ms_to_alpha(attack_ms.squeeze(-1), self.sample_rate),
            ms_to_alpha(release_ms.squeeze(-1), self.sample_rate)
        ], dim=-1).squeeze(-2)
        g_c_smooth = self.smooth_filter(g_c.squeeze(-2), alpha).unsqueeze(-2)

        g_s = g_c_smooth 
        
        g_lin = 10 ** (g_s / 20.0)
        y = x * g_lin
        
        y = y.view(bs, chs, seq_len)
        return y

