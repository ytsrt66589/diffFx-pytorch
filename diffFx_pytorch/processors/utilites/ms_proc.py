import torch 
import torch.nn as nn 
from typing import Dict, Union

class MidSideProc(nn.Module):
    def __init__(self, proc):
        super().__init__()
        self.proc = proc
        
    def forward(self, 
        x: torch.Tensor, 
        mult: float = 0.5, 
        norm_mid_params: Union[Dict[str, torch.Tensor], None] = None,
        norm_side_params: Union[Dict[str, torch.Tensor], None] = None,
        dsp_mid_params: Union[Dict[str, torch.Tensor], None] = None,
        dsp_side_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        left, right = torch.split(x, (1, 1), -2)
        mid, side = left + right, left - right
        mid = mid * mult
        side = side * mult

        mid = self.proc(mid, norm_params=norm_mid_params, dsp_params=dsp_mid_params)
        side = self.proc(side, norm_params=norm_side_params, dsp_params=dsp_side_params)

        left, right = mid + side, mid - side
        y = torch.cat([left, right], -2)

        y = y * mult
        return y
    
    def __str__(self):
        return f'MidSideProc(proc={self.proc})'
    
    def __repr__(self):
        return self.__str__()
    
    def count_num_parameters(self):
        return self.proc.count_num_parameters()