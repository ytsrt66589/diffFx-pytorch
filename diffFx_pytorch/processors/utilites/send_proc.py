import torch 
import torch.nn as nn 
from typing import Dict, Union


class SendProc(nn.Module):
    """A neural network module that implements a parallel processing (send) effect.
    
    The SendProc applies a processor in parallel to the input signal and mixes the processed
    signal with the original input. This is commonly used in audio processing for effects
    like reverb, delay, or any parallel processing where you want to maintain the original
    signal while adding processed content.
    """
    def __init__(self, proc):
        super().__init__()
        self.proc = proc 
        
    def forward(self, 
        x: torch.Tensor, 
        mult: float = 0.5, 
        nn_params: Union[Dict[str, torch.Tensor], None] = None,
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        processed = self.proc(x, nn_params=nn_params, dsp_params=dsp_params)
        return x * mult + processed
    

