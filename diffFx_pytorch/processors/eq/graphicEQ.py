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
    """Differentiable implementation of a multi-band graphic equalizer.
    
    Implementation is based on the following book: 

    ..  [1] Reiss, Joshua D., and Andrew McPherson. 
            Audio effects: theory, implementation and application. CRC Press, 2014.
    
    This processor implements a parallel bank of peak filters to create a graphic equalizer,
    allowing independent gain control over multiple frequency bands. The implementation 
    supports different frequency spacing schemes including ISO standard frequencies, 
    octave spacing, and third-octave spacing.

    The equalizer uses second-order IIR peak filters for each band with transfer function:
    
    .. math::

        H(z) = \\frac{b_0 + b_1z^{-1} + b_2z^{-2}}{1 + a_1z^{-1} + a_2z^{-2}}

    where coefficients are computed based on:
        - Center frequency of each band
        - Q factor (bandwidth)
        - Gain setting for each band
    
    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        num_bands (int): Number of frequency bands. Defaults to 10.
        q_factors (float): Q factor for band filters. Controls bandwidth. Defaults to None.
        eq_type (str): Frequency spacing scheme. Must be one of:
            - 'iso': ISO standard frequencies (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000 Hz)
            - 'octave': Octave-spaced bands
            - 'third_octave': Third-octave spaced bands
            Defaults to 'octave'.

    Parameters Details:
        band_X_gain_db: Gain for band X (where X is 1 to num_bands)
            - Range: -12.0 to 12.0 dB
            - Controls gain at that frequency band
            - Positive values boost, negative values cut


    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their dB ranges
            - Ensure your network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()

    Examples:
        Basic DSP Usage:
            >>> # Create a 10-band graphic EQ with ISO frequencies
            >>> eq = GraphicEqualizer(
            ...     sample_rate=44100,
            ...     num_bands=10,
            ...     q_factors=2.0,
            ...     eq_type='iso'
            ... )
            >>> # Process audio with dsp parameters
            >>> params = {f'band_{i+1}_gain_db': 6.0 for i in range(10)}  # Boost all bands by 6dB
            >>> output = eq(input_audio, dsp_params=params)

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class GraphicEQController(nn.Module):
            ...     def __init__(self, input_size, num_bands):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 32),
            ...             nn.ReLU(),
            ...             nn.Linear(32, num_bands),
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Initialize controller
            >>> eq = GraphicEqualizer(num_bands=10)
            >>> controller = GraphicEQController(input_size=16, num_bands=10)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = controller(features)
            >>> output = eq(input_audio, norm_params=norm_params)
    """
    def __init__(self, sample_rate=44100, param_range = None, num_bands=10, q_factors=None, eq_type='octave'):
        self.num_bands = num_bands
        super().__init__(sample_rate, param_range)
        self.eq_type = GraphicEQType(eq_type)
        
        if eq_type == 'octave':
            self.R = 2 
        elif eq_type == 'third-octave':
            self.R = 2**(1/3)
        
        if q_factors is None:
            self.band_q = np.sqrt(self.R)/(self.R-1)
        else:
            self.band_q = q_factors  # Constant Q design
        
        # Initialize filters
        self.fixed_frequencies = self._get_frequencies()
        self.band_filters = nn.ModuleList([
            BiquadFilter(
                sample_rate=self.sample_rate,
                filter_type='PK',
                # backend='fsm'
            ) for _ in range(num_bands)
        ])

    def _get_frequencies(self) -> list:
        """Get frequency bands based on equalizer type.
        
        Computes center frequencies for bands based on the selected EQ type:
            - ISO: Uses standard ISO center frequencies
            - Octave: Logarithmically spaced bands, one per octave
            - Third-octave: Logarithmically spaced bands, three per octave
            
        Returns:
            list: Center frequencies in Hz for each band
            
        Raises:
            ValueError: If eq_type is not recognized
        """
        if self.eq_type == GraphicEQType.ISO:
            return [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        elif self.eq_type == GraphicEQType.OCTAVE:
            return np.geomspace(20, 20000, self.num_bands).tolist()
        elif self.eq_type == GraphicEQType.THIRD_OCTAVE:
            return np.geomspace(20, 20000, self.num_bands * 3).tolist()
        else:
            raise ValueError(f"Unknown EQ type: {self.eq_type}")
    
    def _register_default_parameters(self):
        """Register gain parameters for each frequency band.
        
        Creates a gain parameter for each band with range -12 dB to +12 dB.
        Parameter names are formatted as 'band_X_gain_db' where X is the band number
        starting from 1.
        """
        self.params = {}
        for i in range(self.num_bands):
            self.params[f'band_{i+1}_gain_db'] = EffectParam(min_val=-12.0, max_val=12.0)
    
    def _prepare_band_parameters(self, 
        band_idx: int, 
        params: Dict[str, torch.Tensor], 
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Prepare filter parameters for a single frequency band.
        
        Args:
            band_idx (int): Index of the band to prepare parameters for
            params (Dict[str, torch.Tensor]): All EQ parameters
            device (torch.device): Device to place tensors on
            
        Returns:
            Dict[str, torch.Tensor]: Parameters for the band's peak filter:
                - gain_db: Gain in dB
                - frequency: Center frequency in Hz
                - q_factor: Q factor for bandwidth
                
        Note:
            Expands scalar parameters to match batch size of provided parameters.
        """
        band_name = f'band_{band_idx+1}'
        freq = torch.tensor(self.fixed_frequencies[band_idx], device=device)
        q = torch.tensor(self.band_q, device=device)
        
        # Expand parameters to match batch size if needed
        batch_size = params[f'{band_name}_gain_db'].shape[0]
        freq = freq.expand(batch_size).float()
        q = q.expand(batch_size).float()
        
        return {
            'gain_db': params[f'{band_name}_gain_db'],
            'frequency': freq,
            'q_factor': q
        }
    
    def process(self, x: torch.Tensor, norm_params: Union[Dict[str, torch.Tensor], None] = None, dsp_params: Union[Dict[str, torch.Tensor], None] = None):
        """Process input signal through the graphic equalizer.

        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
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
                If provided, norm_params must be None.

        Returns:
            torch.Tensor: Processed audio tensor of same shape as input. Shape: (batch, channels, samples)
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
            band_output = self.band_filters[i](x, None, dsp_params=band_params)
            outputs.append(band_output)
            
        # Sum all band outputs and normalize
        output = torch.stack(outputs).sum(dim=0) / self.num_bands
        
        return output
    
    @property
    def frequencies(self) -> list:
        """Get the list of center frequencies.
        
        Returns:
            list: Center frequencies in Hz for all bands
            
        Note:
            Frequencies depend on the equalizer type ('iso', 'octave', or 'third_octave')
            and remain fixed after initialization.
        """
        return self.fixed_frequencies
    

