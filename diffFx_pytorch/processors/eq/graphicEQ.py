import torch 
import torch.nn as nn
import numpy as np 
from enum import Enum
from typing import Dict, Union
from ..base_utils import check_params
from ..base import ProcessorsBase, EffectParam
from ..filters import BiquadFilter


class GraphicEQType(Enum):
    ISO = 'iso'          # ISO standard frequencies
    OCTAVE = 'octave'    # Octave spacing
    THIRD_OCTAVE = 'third_octave'  # 1/3 octave spacing
    

class GraphicEqualizer(ProcessorsBase):
    """Technically correct differentiable implementation of a multi-band graphic equalizer.
    
    Implementation is based on:
    [1] Reiss, Joshua D., and Andrew McPherson. 
        Audio effects: theory, implementation and application. CRC Press, 2014.
    [2] Zölzer, Udo. "DAFX: digital audio effects." John Wiley & Sons, 2011.
    
    This processor implements a series (cascaded) bank of peak filters to create a graphic 
    equalizer. Each band filter processes the signal sequentially, allowing independent 
    gain control over multiple frequency bands with minimal interaction.

    The equalizer uses second-order IIR peak filters in series configuration:
    
    .. math::

        H_{total}(z) = \\prod_{i=1}^{N} H_i(z)

    where each H_i(z) is a peak filter with transfer function:
    
    .. math::

        H_i(z) = \\frac{b_{i0} + b_{i1}z^{-1} + b_{i2}z^{-2}}{1 + a_{i1}z^{-1} + a_{i2}z^{-2}}
    """
    
    def __init__(self, sample_rate=44100, param_range=None, num_bands=10, 
                 eq_type='third_octave', custom_frequencies=None, custom_q_factors=None):
        self.num_bands = num_bands
        self.eq_type = GraphicEQType(eq_type)
        super().__init__(sample_rate, param_range)
        
        self.fixed_frequencies = self._get_frequencies(custom_frequencies)
        self.q_factors = self._get_q_factors(custom_q_factors)
        
        self.band_filters = nn.ModuleList([
            BiquadFilter(
                sample_rate=self.sample_rate,
                filter_type='PK',  # Peak filter
            ) for _ in range(num_bands)
        ])

    def _get_frequencies(self, custom_frequencies=None) -> list:
        """Get frequency bands based on equalizer type."""
        if custom_frequencies is not None:
            return custom_frequencies
            
        if self.eq_type == GraphicEQType.ISO:
            iso_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
            if self.num_bands <= len(iso_freqs):
                return iso_freqs[:self.num_bands]
            else:
                extra_bands = self.num_bands - len(iso_freqs)
                high_freqs = np.geomspace(16000, 20000, extra_bands + 1)[1:]
                return iso_freqs + high_freqs.tolist()
                
        elif self.eq_type == GraphicEQType.OCTAVE:
            base_freq = 31.5
            return [base_freq * (2 ** i) for i in range(self.num_bands)]
            
        elif self.eq_type == GraphicEQType.THIRD_OCTAVE:
            base_freq = 25 
            ratio = 2**(1/3)
            return [base_freq * (ratio ** i) for i in range(self.num_bands)]
        else:
            raise ValueError(f"Unknown EQ type: {self.eq_type}")
    
    def _get_q_factors(self, custom_q_factors=None) -> Union[float, list]:
        """Get Q factors based on equalizer type and spacing."""
        if custom_q_factors is not None:
            return custom_q_factors
            
        if self.eq_type == GraphicEQType.ISO:
            return 4.32
            
        elif self.eq_type == GraphicEQType.OCTAVE:
            return 1.414
            
        elif self.eq_type == GraphicEQType.THIRD_OCTAVE:
            return 4.318
        else:
            return 1.0  
    
    def _register_default_parameters(self):
        """Register gain parameters for each frequency band."""
        self.params = {}
        for i in range(self.num_bands):
            self.params[f'band_{i+1}_gain_db'] = EffectParam(min_val=-12.0, max_val=12.0)
    
    def _prepare_band_parameters(self, 
        band_idx: int, 
        params: Dict[str, torch.Tensor], 
        device: torch.device,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare filter parameters for a single frequency band."""
        band_name = f'band_{band_idx+1}'
        
        freq = torch.tensor(self.fixed_frequencies[band_idx], device=device)
        
        if isinstance(self.q_factors, (list, np.ndarray)):
            q = torch.tensor(self.q_factors[band_idx], device=device)
        else:
            q = torch.tensor(self.q_factors, device=device)
        
        freq = freq.expand(batch_size).float()
        q = q.expand(batch_size).float()
        
        return {
            'gain_db': params[f'{band_name}_gain_db'],
            'frequency': freq,
            'q_factor': q
        }
    
    def process(self, 
                x: torch.Tensor, 
                nn_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """Process input signal through the graphic equalizer.

        The signal is processed through each band filter in series (cascaded connection).
        This is the standard and correct implementation for graphic equalizers.

        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            nn_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Dictionary with keys 'band_X_gain_db' for X in range(1, num_bands+1)
                Each value should be a tensor of shape (batch_size,)
                Values will be mapped to -12.0 to 12.0 dB
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify band gains as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, nn_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input. Shape: (batch, channels, samples)
        """
        check_params(nn_params, dsp_params)
        
        if nn_params is not None:
            params = self.map_parameters(nn_params)
        else:
            params = dsp_params
        
        batch_size = x.shape[0]
        output = x
        for i in range(self.num_bands):
            band_params = self._prepare_band_parameters(i, params, x.device, batch_size)
            output = self.band_filters[i](output, None, dsp_params=band_params)
            
        return output
    
    @property
    def frequencies(self) -> list:
        """Get the list of center frequencies."""
        return self.fixed_frequencies
