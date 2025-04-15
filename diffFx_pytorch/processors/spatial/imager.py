import torch 
import torch.nn as nn
import numpy as np 
from typing import Dict, Union, Tuple
from ..base_utils import check_params 
from ..base import ProcessorsBase, EffectParam
from ..core.midside import * 
from ..filters import LinkwitzRileyFilter


class StereoImager(ProcessorsBase):
    """Differentiable implementation of a multi-band stereo imaging processor.
    
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

    Args:
        sample_rate (int): Audio sample rate in Hz
        num_bands (int): Number of frequency bands. Defaults to 3.

    Attributes:
        crossovers (nn.ModuleList): List of Linkwitz-Riley crossover filters
        num_bands (int): Number of frequency bands

    Parameters Details:
        For each band i:
            bandX_width: Stereo width control for band X
                - 0.0: Mono (only mid signal)
                - 0.5: Original stereo
                - 1.0: Maximum width (enhanced side signal)

        For each crossover i:
            crossoverX_freq: Crossover frequency between bands X and X+1
                - Frequency range scales with band number
                - Default ranges follow standard mastering crossover points
                - Min frequency doubles for each successive crossover
                - Max frequency is limited to 20kHz

    Note:
        - Input must be stereo (two channels)
        - Uses energy-preserving M/S conversion matrices
        - Linkwitz-Riley crossovers ensure phase coherence
        - Total number of parameters = 2 * num_bands - 1
        - Width controls affect the ratio of mid to side signal per band

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their ranges
            - Ensure your network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()

    Examples:
        Basic DSP Usage:
            >>> # Create a 3-band stereo imager
            >>> imager = StereoImager(
            ...     sample_rate=44100,
            ...     num_bands=3
            ... )
            >>> # Process with different width for each band
            >>> output = imager(input_audio, dsp_params={
            ...     'band0_width': 0.3,  # Reduce width in low frequencies
            ...     'band1_width': 0.5,  # Keep mids unchanged
            ...     'band2_width': 0.8,  # Enhance width in highs
            ...     'crossover0_freq': 200.0,  # Low/mid crossover
            ...     'crossover1_freq': 2000.0  # Mid/high crossover
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class ImagerController(nn.Module):
            ...     def __init__(self, input_size, num_params):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, num_params),
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Initialize controller
            >>> imager = StereoImager(num_bands=3)
            >>> num_params = imager.count_num_parameters()  # 5 parameters for 3 bands
            >>> controller = ImagerController(input_size=16, num_params=num_params)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = controller(features)
            >>> output = imager(input_audio, norm_params=norm_params)
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
            
        Note:
            Scales mid and side signals to maintain constant energy across width settings
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