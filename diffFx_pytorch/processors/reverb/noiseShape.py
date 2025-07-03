import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import scipy.signal
from typing import Dict, List, Tuple, Union
from functools import partial
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params
from ..core.fir import FIRConvolution


def octave_band_filterbank(num_taps: int, sample_rate: float):
    bands = [
        31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000,
    ]
    num_bands = len(bands) + 2
    filts = []
    # lowest band is a lowpass
    filt = scipy.signal.firwin(num_taps, 12, fs=sample_rate)
    filt = torch.from_numpy(filt.astype("float32"))
    filt = torch.flip(filt, dims=[0])
    filts.append(filt)
    for fc in bands:
        f_min = fc / np.sqrt(2)
        f_max = fc * np.sqrt(2)
        f_max = np.clip(f_max, a_min=0, a_max=(sample_rate / 2) * 0.999)
        filt = scipy.signal.firwin(
            num_taps, [f_min, f_max], fs=sample_rate, pass_zero=False
        )
        filt = torch.from_numpy(filt.astype("float32"))
        filt = torch.flip(filt, dims=[0])
        filts.append(filt)
    # highest is a highpass
    filt = scipy.signal.firwin(num_taps, 18000, fs=sample_rate, pass_zero=False)
    filt = torch.from_numpy(filt.astype("float32"))
    filt = torch.flip(filt, dims=[0])
    filts.append(filt)
    filts = torch.stack(filts, dim=0)  # (num_bands, num_taps)
    filts = filts.unsqueeze(1)  # (num_bands, 1, num_taps)
    return filts


class NoiseShapedReverb(ProcessorsBase):
    """Differentiable implementation of noise-shaped reverberation.
    
    This processor implements artificial reverberation using frequency-band noise shaping,
    based on the approach proposed in [1]. The method leverages the well-known idea that
    a room impulse response (RIR) can be modeled as the direct sound, a set of early 
    reflections, and a decaying noise-like tail [2].
    
    Implementation is based on: 
    
    ..  [1] Steinmetz, Christian J., Vamsi Krishna Ithapu, and Paul Calamia.
            "Filtered noise shaping for time domain room impulse response estimation from reverberant speech."
            2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2021.
    
    ..  [2] Moorer, James A.
            "About this reverberation business."
            Computer Music Journal (1979): 13-28.
    
    Processing Chain:
        1. Split input into octave bands using filterbank
        2. Generate white noise for each band
        3. Apply band-specific decay envelopes
        4. Sum bands to create impulse response
        5. Convolve with input signal
        6. Mix with dry signal
    
    The transfer function for each band i is:
    
    .. math::
        
        y_i(t) = g_i \\cdot e^{-d_i t} \\cdot n_i(t) * x(t)
        
        y(t) = (1 - mix) \\cdot x(t) + mix \\cdot \\sum_{i=0}^{11} y_i(t)
    
    where:
        - g_i: Band gain (0-1)
        - d_i: Band decay rate
        - n_i(t): Bandpass filtered white noise
        - mix: Wet/dry balance
    
    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        num_samples (int): Number of samples for IR generation. Defaults to 65536.
        num_bandpass_taps (int): Number of filter taps for bandpass filters. Must be odd. Defaults to 1023.
    
    Attributes:
        sample_rate (int): Audio sample rate in Hz
        num_samples (int): IR length in samples
        num_bandpass_taps (int): Bandpass filter length
        num_bands (int): Number of octave bands (12)
    
    Parameters Details:
        band0_gain to band11_gain: Gain for each octave band
            - Range: 0.0 to 1.0
            - Controls the level of each frequency band in the reverb
            - Band 0: 31.5 Hz, Band 11: 16 kHz
            
        band0_decay to band11_decay: Decay rate for each octave band
            - Range: 0.0 to 1.0
            - Controls how quickly each band decays
            - Higher values = faster decay
            
        mix: Wet/dry balance
            - Range: 0.0 to 1.0
            - 0.0: Only dry signal
            - 1.0: Only reverberated signal
    
    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to ranges
            - Ensure network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()
    
    Examples:
        Basic DSP Usage:
            >>> # Create noise-shaped reverb
            >>> reverb = NoiseShapedReverb(
            ...     sample_rate=44100,
            ...     num_samples=65536
            ... )
            >>> # Process with musical settings
            >>> output = reverb(input_audio, dsp_params={
            ...     'band0_gain': 0.8,
            ...     'band1_gain': 0.8,
            ...     'band2_gain': 0.8,
            ...     'band3_gain': 0.8,
            ...     'band4_gain': 0.8,
            ...     'band5_gain': 0.8,
            ...     'band6_gain': 0.8,
            ...     'band7_gain': 0.8,
            ...     'band8_gain': 0.8,
            ...     'band9_gain': 0.8,
            ...     'band10_gain': 0.8,
            ...     'band11_gain': 0.8,
            ...     'band0_decay': 0.3,
            ...     'band1_decay': 0.3,
            ...     'band2_decay': 0.3,
            ...     'band3_decay': 0.3,
            ...     'band4_decay': 0.3,
            ...     'band5_decay': 0.3,
            ...     'band6_decay': 0.3,
            ...     'band7_decay': 0.3,
            ...     'band8_decay': 0.3,
            ...     'band9_decay': 0.3,
            ...     'band10_decay': 0.3,
            ...     'band11_decay': 0.3,
            ...     'mix': 0.3
            ... })
    
        Neural Network Control:
            >>> # Simple parameter prediction
            >>> class ReverbController(nn.Module):
            ...     def __init__(self, input_size, num_params):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(input_size, 64),
            ...             nn.ReLU(),
            ...             nn.Linear(64, num_params),
            ...             nn.Sigmoid()  # Ensures output is in [0,1] range
            ...         )
            ...     
            ...     def forward(self, x):
            ...         return self.net(x)
            >>> 
            >>> # Initialize controller
            >>> reverb = NoiseShapedReverb(sample_rate=44100)
            >>> num_params = reverb.count_num_parameters()  # 25 parameters
            >>> controller = ReverbController(input_size=16, num_params=num_params)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = controller(features)
            >>> output = reverb(input_audio, norm_params=norm_params)
    """
    
    def __init__(self, sample_rate: int = 44100, num_samples: int = 65536, 
                 num_bandpass_taps: int = 1023):
        """Initialize the noise-shaped reverb processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            num_samples: Number of samples for IR generation
            num_bandpass_taps: Number of filter taps for bandpass filters (must be odd)
        """
        self.num_samples = num_samples
        self.num_bandpass_taps = num_bandpass_taps
        self.num_bands = 12  # Now 12 bands: lowpass, 10 bandpass, highpass
        
        super().__init__(sample_rate)
        
        # Create octave band filterbank
        self.register_buffer('filters', octave_band_filterbank(num_bandpass_taps, sample_rate))
        
    def _register_default_parameters(self):
        """Register default parameters for the noise-shaped reverb.
        
        Sets up:
            - band{i}_gain: Gain for each octave band (0.0 to 1.0)
            - band{i}_decay: Decay rate for each octave band (0.0 to 1.0)
            - mix: Wet/dry balance (0.0 to 1.0)
        """
        self.params = {}
        
        # Band gains (12 bands)
        for i in range(self.num_bands):
            self.params[f'band{i}_gain'] = EffectParam(min_val=0.0, max_val=1.0)
        
        # Band decays (12 bands)
        for i in range(self.num_bands):
            self.params[f'band{i}_decay'] = EffectParam(min_val=0.0, max_val=1.0)
        
        # Mix parameter
        self.params['mix'] = EffectParam(min_val=0.0, max_val=1.0)
    
    def process(
        self, 
        x: torch.Tensor, 
        norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, torch.Tensor], None] = None
    ) -> torch.Tensor:
        """Process input signal through noise-shaped reverberation.
        
        Args:
            x: Input audio tensor. Shape: (batch, channels, samples)
            norm_params: Normalized parameters (0 to 1)
            dsp_params: Direct DSP parameters. If provided, norm_params must be None.
                
        Returns:
            torch.Tensor: Reverberated audio tensor of same shape as input
        """
        check_params(norm_params, dsp_params)
        
        # Get parameters
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
        
        batch_size, num_channels, seq_len = x.shape
        device = x.device
        
        # Ensure stereo processing
        if num_channels == 1:
            x = x.repeat(1, 2, 1)
            num_channels = 2
        
        # Extract parameters
        band_gains = torch.stack([
            params[f'band{i}_gain'] for i in range(self.num_bands)
        ], dim=1)  # (batch, num_bands)
        
        band_decays = torch.stack([
            params[f'band{i}_decay'] for i in range(self.num_bands)
        ], dim=1)  # (batch, num_bands)
        
        mix = params['mix'].view(-1, 1, 1)  # (batch, 1, 1)
        
        # Move filters to device
        filters = self.filters.to(device)  # (num_bands, 1, num_taps)
        
        # Generate white noise for IR generation
        pad_size = self.num_bandpass_taps - 1
        wn = torch.randn(batch_size * 2, self.num_bands, self.num_samples + pad_size, device=device)
        
        
        # Filter white noise with each bandpass filter (depthwise)
        wn_filt = F.conv1d(
            wn,
            filters,  # (num_bands, 1, num_taps)
            groups=self.num_bands,
            # padding=pad_size
        )  # (batch*2, num_bands, num_samples)
        wn_filt = wn_filt.view(batch_size, 2, self.num_bands, self.num_samples)
        
        # Prepare gain/decay/envelope
        band_gains = band_gains.view(batch_size, 1, self.num_bands, 1)  # (batch, 1, num_bands, 1)
        band_decays = band_decays.view(batch_size, 1, self.num_bands, 1)  # (batch, 1, num_bands, 1)
        t = torch.linspace(0, 1, steps=self.num_samples, device=device)  # (num_samples,)
        band_decays = (band_decays * 10.0) + 1.0
        env = torch.exp(-band_decays * t.view(1, 1, 1, -1))  # (batch, 1, num_bands, num_samples)
        wn_filt = wn_filt * env * band_gains  # (batch, 2, num_bands, num_samples)
        
        # Sum signals to create impulse response
        ir = wn_filt.mean(dim=2, keepdim=True)  # (batch, 2, 1, num_samples)
        # ir = ir.squeeze(2)  # (batch, 2, num_samples)
        
        # Convolve input with IR (per batch, per channel)
        x_pad = F.pad(x, (self.num_samples - 1, 0))
        
        # Use vmap for efficient batch convolution
        try:
            # Try using torch.vmap if available (PyTorch 2.0+)
            vconv1d = torch.vmap(partial(F.conv1d, groups=2), in_dims=0)
            y = vconv1d(x_pad, torch.flip(ir, dims=[-1]))
        except AttributeError:
            # Fallback for older PyTorch versions
            y = torch.zeros_like(x)
            for b in range(batch_size):
                y[b] = F.conv1d(
                    x_pad[b:b+1], 
                    torch.flip(ir[b:b+1], dims=[-1]), 
                    groups=2
                )
        
        # Create wet/dry mix
        y = (1 - mix) * x + mix * y
        
        return y
