import torch 
import torch.nn as nn 
from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class EffectParam:
    min_val: float
    max_val: float
    default: float = None


class ProcessorsBase(nn.Module):
    """Base class for differentiable audio effect processors.
    
    This class provides the foundation for implementing audio effects processors with support
    for both normalized (0-1) and direct DSP parameter control. It handles parameter
    registration, validation, and mapping between normalized and DSP value ranges.

    The class supports two parameter interfaces:
        1. Normalized parameters (0-1 range) for neural network control
        2. Direct DSP parameters with actual audio processing values

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        param_range (Dict[str, EffectParam], optional): Optional parameter definitions to override or extend default parameters.
    """
    def __init__(self, sample_rate: int = 44100, param_range: Dict[str, EffectParam] = None):
        """Initialize the processor base.
        
        Args:
            sample_rate: Audio sample rate in Hz
            param_range: Optional parameter definitions to override defaults
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.params: Dict[str, EffectParam] = {}
        self._register_default_parameters()
        if param_range:
            self.params.update(param_range)
            
    def _register_default_parameters(self):
        """Register default parameters for the processor.
        
        Override this method to define processor-specific parameters
        using EffectParam dataclass instances.
        """
        pass
    
    def _tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert parameter tensor to dictionary"""
        assert len(tensor.shape) == 2, "Expected 2D tensor" # Check if tensor is 2D 
        assert tensor.shape[1] == len(self.params), f"Expected {len(self.params)} parameters, got {tensor.shape[1]}"
        return {name: tensor[:, i] for i, name in enumerate(self.params.keys())}
    
    def map_parameters(self, nn_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Maps normalized parameters (0-1) to DSP parameter ranges.
        
        Linear interpolation is used for mapping: dsp_value = min_val + (max_val - min_val) * norm_value
            
        Args:
            nn_params: Dictionary of normalized parameter values
            
        Returns:
            Dictionary of mapped DSP parameter values
        """
        return {
            name: param.min_val + (param.max_val - param.min_val) * nn_params[name]
            for name, param in self.params.items()
        }

    def demap_parameters(self, dsp_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Maps DSP parameters back to normalized range (0-1).
        
        Args:
            dsp_params: Dictionary of DSP parameter values
            
        Returns:
            Dictionary of normalized parameter values
        """
        return {
            name: (dsp_params[name] - param.min_val) / (param.max_val - param.min_val)
            for name, param in self.params.items()
        }
    
    def create_dsp_params_batch(self, 
                              params_dict: Dict[str, float], 
                              batch_size: int = 1, 
                              device: Union[str, torch.device] = 'cpu') -> Dict[str, torch.Tensor]:
        """Creates batched tensor parameters from scalar DSP values."""
        batched_params = {}
        
        for name, value in params_dict.items():
            # Check if parameter exists
            if name not in self.params:
                raise KeyError(f"Parameter '{name}' not registered in effect processor")
                
            param_info = self.params[name]
            
            # Validate parameter range
            if value < param_info.min_val or value > param_info.max_val:
                raise ValueError(
                    f"Parameter '{name}' value {value} is outside valid range "
                    f"[{param_info.min_val}, {param_info.max_val}]"
                )
            
            # Create batched tensor
            batched_params[name] = torch.full(
                (batch_size,), 
                value, 
                device=device, 
                dtype=torch.float32
            )
            
        return batched_params
    
    def forward(
        self, 
        x: torch.Tensor, 
        nn_params: Union[torch.Tensor, None] = None, 
        dsp_params: Union[Dict[str, Union[float, torch.Tensor]], None] = None
    ) -> torch.Tensor:
        """Process input with either normalized or DSP parameters.
        
        Args:
            x: Input audio tensor [batch, channels, samples]
            nn_params: Optional normalized parameters tensor [batch, num_params]
            dsp_params: Optional DSP parameters dictionary
                Each parameter can be:
                - float/int: Single value for all batch items
                - 0D tensor: Single value for all batch items
                - 1D tensor: Batch-specific values
                
        Returns:
            Processed audio tensor [batch, channels, samples]
        """
        batch_size = x.shape[0]
        params_dict, dsp_params_dict = None, None
        
        if nn_params is not None: # 
            assert len(nn_params.shape) == 2, "Expected 2D tensor" # Check if tensor is 2D [b, num_params]
            params_dict = self._tensor_to_dict(nn_params)
        
        if dsp_params is not None:
            # Handle DSP parameters
            dsp_params_dict = {}
            for name, value in dsp_params.items():
                if name not in self.params:
                    raise KeyError(f"Unknown parameter: {name}")
                    
                if isinstance(value, (int, float)):
                    # Convert scalar to batched tensor
                    dsp_params_dict[name] = torch.full((batch_size,), float(value), device=x.device, dtype=torch.float32)
                elif isinstance(value, torch.Tensor):
                    # Validate tensor parameter
                    if value.ndim == 0:  # Scalar tensor
                        dsp_params_dict[name] = value.expand(batch_size)
                    elif value.ndim == 1:  # Batched tensor
                        assert value.shape[0] == batch_size, f"Parameter '{name}' batch size {value.shape[0]} != {batch_size}"
                        dsp_params_dict[name] = value
                    else:
                        raise ValueError(f"Parameter '{name}' has too many dimensions: {value.ndim}")
                    
                    # Ensure parameter is on same device as input
                    if value.device != x.device:
                        dsp_params_dict[name] = dsp_params_dict[name].to(x.device)
                else:
                    raise TypeError(f"Parameter '{name}' has invalid type: {type(value)}")

        # params_dict and dsp_params_dict are now dictionaries of tensors
        # that are either normalized or DSP parameters
        
        return self.process(x, params_dict, dsp_params_dict)
    
    def process(
        self, 
        x: torch.Tensor, 
        nn_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        """Process audio with audio effects.
        
        Args:
            x: Input audio tensor [batch, channels, samples]
            nn_params: Optional dictionary of normalized parameters
            dsp_params: Optional dictionary of DSP parameters
            
        Returns:
            Processed audio tensor [batch, channels, samples]
        """
        raise NotImplementedError
    
    def count_num_parameters(self):
        """Returns the number of effect parameters.
        
        Returns:
            Number of registered parameters
        """
        return len(self.params)
    
