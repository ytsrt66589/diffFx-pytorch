import torch 
import torch.nn as nn
import numpy as np 
from typing import Dict, Union, Tuple
from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..core.midside import * 
from ..filters import LinkwitzRileyFilter

class StereoWidener(ProcessorsBase):
    """Differentiable implementation of mid-side stereo width control.
    
    This processor implements stereo width adjustment using mid-side (M/S) processing,
    allowing continuous control from mono to enhanced stereo width. It operates by
    converting the input to M/S representation, scaling the side signal, and converting
    back to left-right stereo.

    The width control is implemented using the following process:
    
    .. math::

        M_{out} = M_{in} * 2(1 - width)
        
        S_{out} = S_{in} * 2(width)

    where:
        - M is the mid (mono) signal: (L + R) / √2
        - S is the side (difference) signal: (L - R) / √2
        - width is the stereo width control parameter
        - Scaling ensures energy preservation across width settings
    """
    def _register_default_parameters(self):
        """Register the width parameter.
        
        Sets up the width parameter with range:
            - 0.0: Mono (collapse to center)
            - 0.5: No change (original stereo)
            - 1.0: Enhanced stereo (maximum width)
        """
        # 0.0 -> mono 0.5 -> no change 1.0 -> stereo
        self.params = {
            'width': EffectParam(min_val=0.0, max_val=1.0),
        } 
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the stereo widener.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 2, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'width': Stereo width control (0 to 1)
                    0.0: Mono/centered
                    0.5: Original stereo width
                    1.0: Maximum width
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify widener parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor. Shape: (batch, 2, samples)
            
        Raises:
            AssertionError: If input is not stereo (two channels)
        """
        check_params(norm_params, dsp_params)
        
        # get parameters 
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        width = params['width']
        bs, chs, seq_len = x.size()
        assert chs == 2, "Input tensor must have shape (bs, 2, seq_len)"
        
        x_ms = lr_to_ms(x, mult=1/np.sqrt(2)) 
        
        # Split M/S signals
        m, s = torch.split(x_ms, (1, 1), -2)
        
        # Adjust side signal based on width
        # width = 0.0 -> side * 0 = mono
        # width = 0.5 -> side * 1 = original stereo
        # width = 1.0 -> side * 2 = wider stereo
        width = width.view(-1, 1, 1)
        mid = m * (2 * (1 - width)) 
        side = s * (2 * width) 
        
        # Recombine M/S
        x_ms = torch.cat([mid, side], -2)
        x_lr = ms_to_lr(x_ms, mult=1/np.sqrt(2))
        
        return x_lr


class MultiBandStereoWidener(ProcessorsBase):
    """Differentiable implementation of a multi-band stereo widener.
    
    This processor implements frequency-dependent stereo width control using mid-side (M/S) 
    processing combined with Linkwitz-Riley crossover filters. It allows independent width 
    control over multiple frequency bands, enabling precise stereo field manipulation 
    across the frequency spectrum.

    The processor splits the signal into frequency bands using a series of Linkwitz-Riley 
    crossover filters, processes each band's stereo width independently, then recombines 
    the bands.

    Processing Chain:
        1. Convert L/R to M/S representation
        2. Split M/S signals into frequency bands using crossovers
        3. Apply independent width control to each band
        4. Sum processed bands
        5. Convert back to L/R representation

    The width control for each band follows:

    .. math::

        M_{out} = M_{in} * 2(1 - width)
        
        S_{out} = S_{in} * 2(width)

    where:
        - M is the mid (mono) signal for the band
        - S is the side (difference) signal for the band
        - width is the stereo width control parameter for that band
    """
    def __init__(self, sample_rate, param_range: Dict[str, EffectParam]=None, num_bands=3):
        self.num_bands = num_bands
        super().__init__(sample_rate, param_range)
        
        # Create crossover filters
        self.crossovers = nn.ModuleList([
            LinkwitzRileyFilter(sample_rate) 
            for _ in range(num_bands - 1)
        ])
    
    def _register_default_parameters(self):
        """Register parameters for band widths and crossover frequencies.
        
        Sets up:
            - Width control for each frequency band (0.0 to 1.0)
            - Crossover frequencies between bands (frequency ranges scale with band)
        """
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
        """Apply stereo width processing to mid/side signals for a single band.
        
        Args:
            mid (torch.Tensor): Mid signal for the band. Shape: (batch, 1, samples)
            side (torch.Tensor): Side signal for the band. Shape: (batch, 1, samples)
            width (torch.Tensor): Width control parameter. Shape: (batch,)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed (mid, side) signals
        """
        width = width.view(-1, 1, 1)  # Reshape for broadcasting
        return (
            mid * (2 * (1 - width)),  # Scale mid based on width
            side * (2 * width)        # Scale side based on width
        )
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """Process input signal through the multi-band stereo imager.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, 2, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                - 'bandi_width': Width control for band i (0 to 1)
                - 'crossoveri_freq': Frequency between band i and band i+1 (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify imager parameters as:
                - float/int: Single value applied to entire batch
                - 0D tensor: Single value applied to entire batch
                - 1D tensor: Batch of values matching input batch size
                Parameters will be automatically expanded to match batch size
                and moved to input device if necessary.
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed stereo audio tensor. Shape: (batch, 2, samples)
            
        Raises:
            AssertionError: If input is not stereo (two channels)
        """
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
            
        bs, chs, seq_len = x.size()
        assert chs == 2, "Input tensor must have shape (batch_size, 2, seq_len)"
        
        # Convert to mid-side
        x_ms = lr_to_ms(x, mult=1/np.sqrt(2))
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
        x_lr = ms_to_lr(x_ms_new, mult=1/np.sqrt(2))
        
        return x_lr
    
