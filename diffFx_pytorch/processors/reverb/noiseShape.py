import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import scipy.signal
from typing import Dict, List, Tuple, Union
from functools import partial
from ..base import ProcessorsBase, EffectParam
from ..base_utils import check_params


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
    """
    
    def __init__(self, sample_rate: int = 44100, num_samples: int = 65536, 
                 num_bandpass_taps: int = 1023):
        """Initialize the noise-shaped reverb processor."""
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
        
        for i in range(self.num_bands):
            self.params[f'band{i}_gain'] = EffectParam(min_val=0.0, max_val=1.0)
        
        for i in range(self.num_bands):
            self.params[f'band{i}_decay'] = EffectParam(min_val=0.0, max_val=1.0)
        
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
        
        if num_channels == 1:
            x = x.repeat(1, 2, 1)
            num_channels = 2
        
        band_gains = torch.stack([
            params[f'band{i}_gain'] for i in range(self.num_bands)
        ], dim=1)  
        
        band_decays = torch.stack([
            params[f'band{i}_decay'] for i in range(self.num_bands)
        ], dim=1)  
        
        mix = params['mix'].view(-1, 1, 1) 
        

        filters = self.filters.to(device)  
        

        pad_size = self.num_bandpass_taps - 1
        wn = torch.randn(batch_size * 2, self.num_bands, self.num_samples + pad_size, device=device)
        
        wn_filt = F.conv1d(
            wn,
            filters, 
            groups=self.num_bands,
        )  
        wn_filt = wn_filt.view(batch_size, 2, self.num_bands, self.num_samples)
        

        band_gains = band_gains.view(batch_size, 1, self.num_bands, 1)  
        band_decays = band_decays.view(batch_size, 1, self.num_bands, 1)  
        t = torch.linspace(0, 1, steps=self.num_samples, device=device) 
        band_decays = (band_decays * 10.0) + 1.0
        env = torch.exp(-band_decays * t.view(1, 1, 1, -1))  
        wn_filt = wn_filt * env * band_gains 
        
        ir = wn_filt.mean(dim=2, keepdim=True) 

        x_pad = F.pad(x, (self.num_samples - 1, 0))
        
        try:
            vconv1d = torch.vmap(partial(F.conv1d, groups=2), in_dims=0)
            y = vconv1d(x_pad, torch.flip(ir, dims=[-1]))
        except AttributeError:
            y = torch.zeros_like(x)
            for b in range(batch_size):
                y[b] = F.conv1d(
                    x_pad[b:b+1], 
                    torch.flip(ir[b:b+1], dims=[-1]), 
                    groups=2
                )
        
        y = (1 - mix) * x + mix * y
        
        return y
