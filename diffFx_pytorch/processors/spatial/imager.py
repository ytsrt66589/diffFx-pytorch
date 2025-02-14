import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union

from enum import Enum

from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..core.midside import * 
from ..filters import LinkwitzRileyFilter


class Imager(ProcessorsBase):
    def __init__(self, sample_rate, num_bands=3):
        self.num_bands = num_bands
        super().__init__(sample_rate)
        
        # Create crossover filters
        self.crossovers = nn.ModuleList([
            LinkwitzRileyFilter(sample_rate) 
            for _ in range(num_bands - 1)
        ])
    
    def _register_default_parameters(self):
        self.params = {}
        # Width controls for each band (0 = only mid, 1 = only side)
        for i in range(self.num_bands):
            self.params[f'band{i}_width'] = EffectParam(
                min_val=0.0, 
                max_val=1.0
            )
        
        # Crossover frequencies between bands
        # Using standard mastering crossover points as defaults
        for i in range(self.num_bands - 1):
            min_freq = 20.0 * (2 ** i)
            max_freq = min(20000.0, min_freq * 100)
            self.params[f'crossover{i}_freq'] = EffectParam(
                min_val=min_freq, 
                max_val=max_freq
            )
    
    def _apply_width(self, mid: torch.Tensor, side: torch.Tensor, 
                     width: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply stereo width to mid/side signals"""
        width = width.view(-1, 1, 1)  # Reshape for broadcasting
        return (
            mid * (2 * (1 - width)),  # Scale mid based on width
            side * (2 * width)        # Scale side based on width
        )
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
            
        bs, chs, seq_len = x.size()
        assert chs == 2, "Input tensor must have shape (batch_size, 2, seq_len)"
        
        # Convert to mid-side
        x_ms = lr_to_ms(x, mult=np.sqrt(2))
        mid, side = torch.split(x_ms, (1, 1), dim=-2)
        
        # Split into frequency bands using LR crossovers
        mid_bands = []
        side_bands = []
        current_mid = mid
        current_side = side
        
        # Apply crossovers in series
        for i, crossover in enumerate(self.crossovers):
            # Split mid signal
            mid_lh = crossover.process(current_mid, norm_params=None,dsp_params={
                'frequency': params[f'crossover{i}_freq']
            })
            mid_low, mid_high = torch.split(mid_lh, (1,1), dim=-2)
            # Split side signal (using same crossover frequency)
            side_lh = crossover.process(current_side, norm_params=None,dsp_params={
                'frequency': params[f'crossover{i}_freq']
            })
            side_low, side_high = torch.split(side_lh, (1,1), dim=-2)
            
            mid_bands.append(mid_low)
            side_bands.append(side_low)
            current_mid = mid_high
            current_side = side_high
        
        # Add the final high bands
        mid_bands.append(current_mid)
        side_bands.append(current_side)
        
        # Process each band
        processed_mid = torch.zeros_like(mid)
        processed_side = torch.zeros_like(side)
        
        for i in range(self.num_bands):
            # Apply width processing to each band
            width = params[f'band{i}_width']
            mid_processed, side_processed = self._apply_width(
                mid_bands[i], 
                side_bands[i], 
                width
            )
            
            # Sum the processed bands
            processed_mid += mid_processed
            processed_side += side_processed
        
        # Combine processed mid-side signals
        x_ms_new = torch.cat([processed_mid, processed_side], dim=-2)
        
        # Convert back to left-right
        x_lr = ms_to_lr(x_ms_new)
        
        return x_lr