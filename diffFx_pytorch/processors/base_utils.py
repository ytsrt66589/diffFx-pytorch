import torch 
from typing import Dict, Union

def check_params(norm_params, dsp_params) -> None:
    if norm_params is None and dsp_params is None:
        raise ValueError("Either norm_params or dsp_params must be provided")
    
    if norm_params is not None and dsp_params is not None:
        raise ValueError("Cannot provide both norm_params and dsp_params simultaneously")

def create_dsp_params_batch(params_dict: Dict[str, float], batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    return {
        key: torch.full((batch_size,), value, device=device)
        for key, value in params_dict.items()
    }