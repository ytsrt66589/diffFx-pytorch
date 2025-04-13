import torch 
import torch.nn as nn 
from typing import Dict, Union


class SendProc(nn.Module):
    def __init__(self, proc):
        super().__init__()
        self.proc = proc 
        
    def forward(self, 
        x: torch.Tensor, 
        mult: float = 0.5, 
        norm_params: Union[Dict[str, torch.Tensor], None] = None,
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        processed = self.proc(x, norm_params=norm_params, dsp_params=dsp_params)
        return x * mult + processed
    

