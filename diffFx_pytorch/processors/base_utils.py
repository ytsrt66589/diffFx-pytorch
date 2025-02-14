import torch 
from typing import Dict, Union

def check_params(norm_params, dsp_params) -> None:
    """
    Check parameters validity:
    1. At least one parameter should not be None
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

'''
    dsp_params: {
        'gain': torch.tensor([0.0]),
        'frequency': torch.tensor([1000.0]),
        'q_factor': torch.tensor([0.707]),
    }
'''
def map_dict_dsp_params_to_tensor(dsp_params: Dict[str, torch.Tensor], processor) -> torch.Tensor:
    """
    Map dictionary of DSP parameters to tensor
    
    Args:
        dsp_params: Dictionary of DSP parameters
        processor: Processor object
        
    Returns:
        Tensor of DSP parameters
    """
    # Get ordered parameter names from processor
    param_names = list(processor.params.keys())
    
    # Create list to store parameters in correct order
    ordered_params = []
    
    # Map each parameter in the correct order
    for param_name in param_names:
        if param_name not in dsp_params:
            raise KeyError(f"Missing parameter {param_name} in dsp_params")
        ordered_params.append(dsp_params[param_name])
    
    # Concatenate all parameters into single tensor
    return torch.cat(ordered_params, dim=-1)