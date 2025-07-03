import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, Union
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from .biquad import BiquadFilter


class ButterworthFilter(ProcessorsBase):
    """
    Butterworth filter implementation using cascaded BiquadFilter processors.
    
    This implements a Butterworth filter by cascading multiple biquad sections,
    each with the correct Q factor based on pole locations.
    
    Args:
        sample_rate (int): Sample rate in Hz
        order (int): Filter order (must be even)
        filter_type (str): 'lowpass' or 'highpass'
        **kwargs: Additional arguments passed to BiquadFilter
    """
    
    def __init__(self, sample_rate=44100, order=4, filter_type='lowpass', **kwargs):
        super().__init__(sample_rate)
        
        if order % 2 != 0:
            raise ValueError("Order must be even for biquad implementation")
            
        self.order = order
        self.filter_type = filter_type.lower()
        self.num_biquads = order // 2
        
        # Pre-calculate Q factors for each biquad stage (Butterworth pole locations)
        self.register_buffer('q_factors', self._calculate_butterworth_q_factors())
        
        # Create biquad processors
        biquad_type = 'LP' if filter_type.lower() == 'lowpass' else 'HP'
        self.biquads = nn.ModuleList([
            BiquadFilter(
                sample_rate=sample_rate,
                filter_type=biquad_type,
                **kwargs
            )
            for _ in range(self.num_biquads)
        ])
        
    def _calculate_butterworth_q_factors(self) -> torch.Tensor:
        """
        Calculate Q factors for each biquad stage in a Butterworth filter.
        
        For Butterworth filters, Q_k = 1 / (2 * cos(theta_k))
        where theta_k = (2k-1) * pi / (2*N)
        """
        q_factors = []
        
        for k in range(1, self.num_biquads + 1):
            theta_k = (2*k - 1) * math.pi / (2 * self.order)
            q_k = 1.0 / (2.0 * math.cos(theta_k))
            q_factors.append(q_k)
            
        return torch.tensor(q_factors, dtype=torch.float32)
    
    def _register_default_parameters(self):
        """Register filter parameters."""
        self.params = {
            'frequency': EffectParam(min_val=20.0, max_val=20000.0),  # Cutoff frequency
        }
    
    def process(self, x: torch.Tensor, 
                norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """
        Process input signal through the Butterworth filter.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
            dsp_params (Dict[str, torch.Tensor], optional): Direct DSP parameters.
                If provided, norm_params must be None.
                
        Returns:
            torch.Tensor: Filtered audio tensor of same shape as input
        """
        check_params(norm_params, dsp_params)
        
        # Get cutoff frequency
        if dsp_params is not None:
            frequency = dsp_params['frequency']
        else:
            params = self.map_parameters(norm_params)
            frequency = params['frequency']
        
        # Ensure frequency is properly shaped
        if not isinstance(frequency, torch.Tensor):
            frequency = torch.tensor(frequency, device=x.device, dtype=x.dtype)
        if frequency.dim() == 0:
            frequency = frequency.expand(x.shape[0])
            
        output = x
        
        # Process through each biquad stage with correct Q factor
        for i, biquad in enumerate(self.biquads):
            # Get Q factor for this stage
            q_factor = self.q_factors[i].expand_as(frequency)
            
            # Create parameter dictionary for this biquad
            biquad_params = {
                'frequency': frequency,
                'q_factor': q_factor,
                'gain_db': torch.zeros_like(frequency)  # 0 dB gain for Butterworth
            }
            
            # Apply biquad filter
            output = biquad.process(output, dsp_params=biquad_params)
            
        return output
    
    def get_biquad_parameters(self, cutoff_frequency: float) -> List[Dict[str, float]]:
        """
        Get the parameters for each biquad stage for analysis/debugging.
        
        Args:
            cutoff_frequency (float): Cutoff frequency in Hz
            
        Returns:
            List of parameter dictionaries for each biquad
        """
        params = []
        
        for i in range(self.num_biquads):
            theta_k = (2*(i+1) - 1) * math.pi / (2 * self.order)
            q_factor = 1.0 / (2.0 * math.cos(theta_k))
            
            params.append({
                'stage': i + 1,
                'frequency': cutoff_frequency,
                'q_factor': q_factor,
                'gain_db': 0.0,
                'theta_rad': theta_k,
                'theta_deg': theta_k * 180 / math.pi
            })
            
        return params


class LinkwitzRileyFilter(ProcessorsBase):
    """
    Linkwitz-Riley filter implementation based on reference analysis.
    
    Key insight from reference:
    1. Create Butterworth filter of order N/2
    2. Duplicate poles and zeros (equivalent to cascading)  
    3. Square the gain
    4. Convert back to filter coefficients
    
    This is mathematically equivalent to cascading the same Butterworth filter twice.
    """
    
    def __init__(self, sample_rate=44100, order=4, **kwargs):
        super().__init__(sample_rate)
        
        if order % 2 != 0:
            raise ValueError("Linkwitz-Riley filter order must be even")
        
        self.order = order
        self.butterworth_order = order // 2
        
        # Create Butterworth filters that will be applied twice
        self.lowpass_butter = ButterworthFilter(
            sample_rate=sample_rate,
            order=self.butterworth_order,
            filter_type='lowpass',
            **kwargs
        )
        
        self.highpass_butter = ButterworthFilter(
            sample_rate=sample_rate,
            order=self.butterworth_order,
            filter_type='highpass',
            **kwargs
        )
        
    def _register_default_parameters(self):
        self.params = {
            'frequency': EffectParam(min_val=20.0, max_val=20000.0),
        }
    
    def process(self, x: torch.Tensor, 
                norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """
        Process using the reference approach: cascade same filter twice.
        """
        check_params(norm_params, dsp_params)
        
        # Lowpass path: apply Butterworth LP twice (cascaded)
        if dsp_params is not None:
            low = self.lowpass_butter.process(x, dsp_params=dsp_params)
            low = self.lowpass_butter.process(low, dsp_params=dsp_params)
        else:
            low = self.lowpass_butter.process(x, norm_params=norm_params)
            low = self.lowpass_butter.process(low, norm_params=norm_params)
        
        # Highpass path: apply Butterworth HP twice (cascaded)  
        if dsp_params is not None:
            high = self.highpass_butter.process(x, dsp_params=dsp_params)
            high = self.highpass_butter.process(high, dsp_params=dsp_params)
        else:
            high = self.highpass_butter.process(x, norm_params=norm_params)
            high = self.highpass_butter.process(high, norm_params=norm_params)
        
        return torch.cat((low, high), dim=1)
    
    def get_separate_outputs(self, x: torch.Tensor, 
                           norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                           dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate lowpass and highpass outputs."""
        check_params(norm_params, dsp_params)
        
        # Lowpass: Butterworth applied twice
        if dsp_params is not None:
            low = self.lowpass_butter.process(x, dsp_params=dsp_params)
            low = self.lowpass_butter.process(low, dsp_params=dsp_params)
        else:
            low = self.lowpass_butter.process(x, norm_params=norm_params)
            low = self.lowpass_butter.process(low, norm_params=norm_params)
        
        # Highpass: Butterworth applied twice
        if dsp_params is not None:
            high = self.highpass_butter.process(x, dsp_params=dsp_params)
            high = self.highpass_butter.process(high, dsp_params=dsp_params)
        else:
            high = self.highpass_butter.process(x, norm_params=norm_params)
            high = self.highpass_butter.process(high, norm_params=norm_params)
        
        return low, high
    
    