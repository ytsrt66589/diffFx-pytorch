import torch 
import torch.nn as nn 
from typing import Dict, Union


class SendProc(nn.Module):
    """A neural network module that implements a parallel processing (send) effect.
    
    The SendProc applies a processor in parallel to the input signal and mixes the processed
    signal with the original input. This is commonly used in audio processing for effects
    like reverb, delay, or any parallel processing where you want to maintain the original
    signal while adding processed content.
    
    Args:
        proc (nn.Module): The processor module to be applied in parallel.
                         This processor should accept norm_params and dsp_params as arguments.
    
    Forward Args:
        x (torch.Tensor): Input signal of shape [batch_size, channels, num_samples].
        mult (float, optional): Mixing multiplier for the processed signal. Defaults to 0.5.
        norm_params (Dict[str, torch.Tensor], optional): Normalization parameters for the processor.
        dsp_params (Dict[str, torch.Tensor], optional): DSP parameters for the processor.
    
    Returns:
        torch.Tensor: Mixed signal of the same shape as input.
    
    Example:
        >>> # 1. Create a controller for parameter prediction
        >>> class Controller(nn.Module):
        ...     def __init__(self, input_size):
        ...         super().__init__()
        ...         self.net = nn.Sequential(
        ...             nn.Linear(input_size, 32),
        ...             nn.ReLU(),
        ...             nn.Linear(32, 1),  # Single parameter
        ...             nn.Sigmoid()  # Ensures output is in [0,1] range
        ...         )
        ...     
        ...     def forward(self, x):
        ...         return self.net(x)
        >>> 
        >>> # 2. Create Send processor with a processor
        >>> controller = Controller(input_size=16)
        >>> proc = SomeProcessor(sample_rate=44100)
        >>> send_processor = SendProc(proc)
        >>> 
        >>> # 3. Process with features
        >>> batch_size = 4
        >>> features = torch.randn(batch_size, 16)
        >>> norm_params = controller(features)
        >>> input_audio = torch.randn(batch_size, 2, 44100)
        >>> output = send_processor(
        ...     input_audio,
        ...     norm_params=norm_params
        ... )
    """
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
    

