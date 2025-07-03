import torch 
from typing import Dict, Union

def check_params(norm_params, dsp_params) -> None:
    """
    Check parameters validity:
    1. At least one parameter (norm_params or dsp_params) should not be None
    2. Both parameters cannot be not None simultaneously
    
    Args:
        norm_params: Dictionary of normalized parameters or None
        dsp_params: Dictionary of DSP parameters or None
        
    Raises:
        ValueError: If both parameters are None or both are not None
    """
    if norm_params is None and dsp_params is None:
        raise ValueError("Either norm_params or dsp_params must be provided")
    
    if norm_params is not None and dsp_params is not None:
        raise ValueError("Cannot provide both norm_params and dsp_params simultaneously")

def create_dsp_params_batch(params_dict: Dict[str, float], batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Convert scalar DSP parameters to batched tensor parameters.
    
    Args:
        params_dict: Dictionary of parameter names and their scalar values
        batch_size: Number of copies to create in batch
        device: Target device for tensors
        
    Returns:
        Dictionary of parameter names and their batched tensor values
    """
    return {
        key: torch.full((batch_size,), value, device=device)
        for key, value in params_dict.items()
    }