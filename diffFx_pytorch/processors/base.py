import torch 
import torch.nn as nn 
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from .base_utils import check_params

@dataclass
class EffectParam:
    min_val: float
    max_val: float
    default: float = None
    

class ProcessorsBase(nn.Module):
    def __init__(self, sample_rate: int = 44100, param_range: Dict[str, EffectParam] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.params: Dict[str, EffectParam] = {}
        self._register_default_parameters()
        if param_range:
            self.params.update(param_range)
            
    def _register_default_parameters(self):
        """Override to set default parameters"""
        pass
    
    def _tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert parameter tensor to dictionary"""
        assert len(tensor.shape) == 2, "Expected 2D tensor" # Check if tensor is 2D 
        assert tensor.shape[1] == len(self.params), f"Expected {len(self.params)} parameters, got {tensor.shape[1]}"
        return {name: tensor[:, i] for i, name in enumerate(self.params.keys())}
    
    def map_parameters(self, norm_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Maps normalized parameters (0~1) to actual DSP values"""
        return {
            name: param.min_val + (param.max_val - param.min_val) * norm_params[name]
            for name, param in self.params.items()
        }

    def demap_parameters(self, dsp_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Maps actual DSP values back to normalized parameters (0~1)
        
        Args:
            dsp_params: Dictionary of parameter names to their DSP values
            
        Returns:
            Dictionary of parameter names to their normalized values (0~1)
        """
        return {
            name: (dsp_params[name] - param.min_val) / (param.max_val - param.min_val)
            for name, param in self.params.items()
        }
    
    def forward(self, x: torch.Tensor, norm_params: Union[torch.Tensor, None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        check_params(norm_params, dsp_params)
        params_dict, dsp_params_dict = None, None
        if norm_params is not None:
            assert len(norm_params.shape) == 2, "Expected 2D tensor" # Check if tensor is 2D [b, num_params]
            params_dict = self._tensor_to_dict(norm_params)
        if dsp_params is not None:
            # assert len(dsp_params.shape) == 2, "Expected 2D tensor" # Check if tensor is 2D [b, num_params]
            dsp_params_dict = dsp_params #self._tensor_to_dict(dsp_params)
        return self.process(x, params_dict, dsp_params_dict)
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
            dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """Override this to implement effect processing"""
        raise NotImplementedError
    
    def count_num_parameters(self):
        return len(self.params)