import torch 
import torch.nn as nn 
from typing import Dict, Union

class MidSideProc(nn.Module):
    """A neural network module that processes audio signals using the Mid-Side (M/S) technique.
    
    The Mid-Side processing technique splits a stereo signal into mid (sum) and side (difference)
    components, processes them separately, and then recombines them back into stereo.
    This allows for independent processing of the center and stereo information in the mix.
    
    Args:
        proc (nn.Module): The processor module to be applied to both mid and side channels.
                         This processor should accept norm_params and dsp_params as arguments.
    
    Forward Args:
        x (torch.Tensor): Input stereo signal of shape [batch_size, 2, num_samples].
        mult (float, optional): Multiplier for the mid/side conversion. Defaults to 0.5.
        norm_mid_params (Dict[str, torch.Tensor], optional): Normalization parameters for mid channel.
        norm_side_params (Dict[str, torch.Tensor], optional): Normalization parameters for side channel.
        dsp_mid_params (Dict[str, torch.Tensor], optional): DSP parameters for mid channel processing.
        dsp_side_params (Dict[str, torch.Tensor], optional): DSP parameters for side channel processing.
    
    Returns:
        torch.Tensor: Processed stereo signal of the same shape as input.
    
    Example:
        >>> # 1. Create a processor for mid and side channels
        >>> class Controller(nn.Module):
            ...     def __init__(self, input_size):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, 1),  # Single gain parameter
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
        >>> 
        >>> # 2. Create MidSide processor with the channel processor
        >>> controller = Controller(input_size=16)
        >>> proc = SomeProcessor(sample_rate=44100)
        >>> ms_processor = MidSideProc(proc)
        >>> 
        >>> # 3. Process with features
        >>> batch_size = 4
        >>> features = torch.randn(batch_size, 16)
        >>> norm_mid_params = controller(features)
        >>> norm_side_params = controller(features)
        >>> stereo_input = torch.randn(batch_size, 2, 44100)
        >>> output = ms_processor(
        ...     stereo_input,
        ...     norm_mid_params=norm_mid_params,
        ...     norm_side_params=norm_side_params
        ... )
    """
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