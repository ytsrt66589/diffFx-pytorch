import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from enum import Enum
from typing import Dict, List, Tuple, Union
from ..base_utils import check_params
from ..base import ProcessorsBase, EffectParam
from ..filters import BiquadFilter


class GraphicEQType(Enum):
    ISO = 'iso'          # ISO standard frequencies
    OCTAVE = 'octave'    # Octave spacing
    THIRD_OCTAVE = 'third_octave'  # 1/3 octave spacing
    
    
class GraphicEqualizer(ProcessorsBase):
    """
    A differentiable graphic equalizer implementation.
    
    Args:
        sample_rate (int): Sampling rate in Hz
        num_bands (int): Number of frequency bands (default: 10)
        eq_type (str): Type of frequency spacing ('iso', 'octave', 'third_octave')
        param_smooth (bool): Whether to apply parameter smoothing (default: True)
    """
    def __init__(self, sample_rate=44100, num_bands=10, q_factors=2.0, eq_type='iso'):
        self.num_bands = num_bands
        super().__init__(sample_rate)
        self.eq_type = GraphicEQType(eq_type)
        self.band_q = q_factors  # Constant Q design
        
        # Initialize filters
        self.fixed_frequencies = self._get_frequencies()
        self.band_filters = nn.ModuleList([
            BiquadFilter(
                sample_rate=self.sample_rate,
                filter_type='PK'
            ) for _ in range(num_bands)
        ])

    def _get_frequencies(self) -> list:
        """Get frequency bands based on EQ type"""
        if self.eq_type == GraphicEQType.ISO:
            return [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        elif self.eq_type == GraphicEQType.OCTAVE:
            return np.geomspace(20, 20000, self.num_bands).tolist()
        elif self.eq_type == GraphicEQType.THIRD_OCTAVE:
            return np.geomspace(20, 20000, self.num_bands * 3).tolist()
        else:
            raise ValueError(f"Unknown EQ type: {self.eq_type}")
    
    def _register_default_parameters(self):
        """Register parameters for each band"""
        self.params = {}
        for i in range(self.num_bands):
            self.params[f'band_{i+1}_gain_db'] = EffectParam(min_val=-12.0, max_val=12.0)
    
    def _prepare_band_parameters(self, band_idx: int, params: Dict[str, torch.Tensor], 
                               device: torch.device) -> Dict[str, torch.Tensor]:
        """Prepare parameters for a single band"""
        band_name = f'band_{band_idx+1}'
        freq = torch.tensor(self.fixed_frequencies[band_idx], device=device)
        q = torch.tensor(self.band_q, device=device)
        
        # Expand parameters to match batch size if needed
        batch_size = params[f'{band_name}_gain_db'].shape[0]
        freq = freq.expand(batch_size)
        q = q.expand(batch_size)
        
        return {
            'gain_db': params[f'{band_name}_gain_db'],
            'frequency': freq,
            'q_factor': q
        }
    
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """
        Process input signal through the graphic equalizer.
        
        Args:
            x (torch.Tensor): Input signal [B x C x T]
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0-1)
            dsp_params (Dict[str, torch.Tensor], optional): Direct DSP parameters
            
        Returns:
            torch.Tensor: Processed signal [B x C x T]
        """
        check_params(norm_params, dsp_params)
        
        # Map parameters
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
                
        # Process each band in parallel
        outputs = []
        for i in range(self.num_bands):
            band_params = self._prepare_band_parameters(i, params, x.device)
            band_output = self.band_filters[i](x, dsp_params=band_params)
            outputs.append(band_output)
            
        # Sum all band outputs and normalize
        output = torch.stack(outputs).sum(dim=0) / self.num_bands
        
        return output
    
    @property
    def frequencies(self) -> list:
        """Get the list of center frequencies"""
        return self.fixed_frequencies
    

# Graphic Equalizer 
# class GraphicEqualizer(ProcessorsBase):
#     def __init__(self, sample_rate, num_bands=10):
#         self.num_bands = num_bands
        
#         super().__init__(sample_rate)
        
#         # Create a list of peaking filters for each band
#         self.band_filters = [
#             BiquadFilter(
#                 sample_rate=self.sample_rate,
#                 filter_type='PK'  # Peaking filter for each band
#             ) for _ in range(num_bands)
#         ]
        
#     def _register_default_parameters(self):
#         # Calculate frequency bands geometrically
#         start_freq = 31.5  # Starting frequency (common in graphic EQs)
#         end_freq = 20000   # Ending frequency
#         frequencies = np.geomspace(start_freq, end_freq, self.num_bands)
        
#         self.params = {}
        
#         # Create parameters for each band
#         for i, freq in enumerate(frequencies):
#             band_name = f'band_{i+1}'
#             self.params.update({
#                 f'{band_name}_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
#                 f'{band_name}_frequency': EffectParam(min_val=freq*0.9, max_val=freq*1.1),
#                 f'{band_name}_q_factor': EffectParam(min_val=1.0, max_val=4.0),
#             })
            
#     def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        
#         check_params(norm_params, dsp_params)
        
#         if norm_params is not None:
#             denorm_params = self.map_parameters(norm_params)
#         else:
#             denorm_params = dsp_params
        
#         x_processed = x
#         # Process through each band filter sequentially
#         for i in range(self.num_bands):
#             band_name = f'band_{i+1}'
            
#             # Get parameters for current band
#             band_params = {
#                 'gain_db': denorm_params[f'{band_name}_gain_db'],
#                 'frequency': denorm_params[f'{band_name}_frequency'],
#                 'q_factor': denorm_params[f'{band_name}_q_factor']
#             }
            
#             # Apply the band filter
#             x_processed = self.band_filters[i](x_processed, dsp_params=band_params)
            
#         return x_processed

