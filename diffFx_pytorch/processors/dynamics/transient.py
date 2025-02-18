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
    def __init__(self, mode="attack", sample_rate=44100):
        super().__init__(sample_rate=sample_rate)
        self.mode = mode
        self.ballistics = Ballistics()
        self.power_filter = TruncatedOnePoleIIRFilter(iir_len=16384)
        
    def _register_default_parameters(self):
        self.params = {
            'power_mem_ms': EffectParam(min_val=0.1, max_val=5.0),
            'fast_attack_ms': EffectParam(min_val=0.01, max_val=5.0),
            'slow_attack_ms': EffectParam(min_val=10.0, max_val=50.0),
            'release_ms': EffectParam(min_val=10.0, max_val=100.0)
        }
        
    def _ms_to_coeff(self, time_ms: torch.Tensor) -> torch.Tensor:
        return torch.exp(-1.0 / (self.sample_rate * time_ms / 1000.0))
        
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], ori_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        params = self.map_parameters(norm_params)
        
        # 1. Normalize
        x_peak = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        x_norm = x / (x_peak + 1e-8)

        # 2. Get coefficients and ensure proper shapes
        z_alpha_power = ms_to_z_alpha(params['power_mem_ms'], self.sample_rate).unsqueeze(-1)  # Shape: (batch, 1)
        g_fast = self._ms_to_coeff(params['fast_attack_ms']).unsqueeze(-1)         # Shape: (batch, 1)
        g_slow = self._ms_to_coeff(params['slow_attack_ms']).unsqueeze(-1)         # Shape: (batch, 1)
        g_release = self._ms_to_coeff(params['release_ms']).unsqueeze(-1)          # Shape: (batch, 1)
        
        # Power Envelope 
        x_squared = x_norm.square().mean(-2)  # Shape: (batch, time)
        
        # Ensure power filter input shapes are correct
        power = self.power_filter(x_squared, z_alpha_power)  # Shape: (batch, time)
        
        # Compute power derivative
        power_deriv = torch.zeros_like(power)
        power_deriv[:, 0] = power[:, 0]
        power_deriv[:, 1:] = power[:, 1:] - power[:, :-1]
        
        # 3. Ballistics processing for fast and slow envelopes
        z_alpha_fast = torch.stack([
            self._ms_to_z_alpha(params['release_ms']),
            self._ms_to_z_alpha(params['fast_attack_ms'])
        ], dim=-1)  # Shape: (batch, 2)
        
        z_alpha_slow = torch.stack([
            self._ms_to_z_alpha(params['release_ms']),
            self._ms_to_z_alpha(params['slow_attack_ms'])
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
