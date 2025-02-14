import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Dict, List, Tuple, Union

from enum import Enum

from ..base_utils import check_params
from ..base import ProcessorsBase, EffectParam
# from ..filters import BiquadFilter
from ..core.iir import IIRFilter

# ref: https://github.com/mod-audio/guitarix/blob/master/trunk/src/faust/tonestack.dsp?fbclid=IwAR02TRPhiyVm5d_K0Df9KR8gxzbYcZX80NPvzT3ciCMn4r0V-iUJg8yH0YU

class TonestackPreset(Enum):
    # Fender Models
    BASSMAN = {  # 59 Bassman 5F6-A
        'R1': 250e3, 'R2': 1e6, 'R3': 25e3, 'R4': 56e3,
        'C1': 250e-12, 'C2': 20e-9, 'C3': 20e-9
    }
    MESA = {  # Mesa Boogie Mark
        'R1': 250e3, 'R2': 250e3, 'R3': 25e3, 'R4': 100e3,
        'C1': 250e-12, 'C2': 100e-9, 'C3': 47e-9
    }
    TWIN = {  # 69 Twin Reverb AA270
        'R1': 250e3, 'R2': 250e3, 'R3': 10e3, 'R4': 100e3,
        'C1': 120e-12, 'C2': 100e-9, 'C3': 47e-9
    }
    PRINCETON = {  # 64 Princeton AA1164
        'R1': 250e3, 'R2': 250e3, 'R3': 4.8e3, 'R4': 100e3,
        'C1': 250e-12, 'C2': 100e-9, 'C3': 47e-9
    }
    
    # Marshall Models
    JCM800 = {  # 59/81 JCM-800 Lead 100 2203
        'R1': 220e3, 'R2': 1e6, 'R3': 22e3, 'R4': 33e3,
        'C1': 470e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    JCM2000 = {  # 81 2000 Lead
        'R1': 250e3, 'R2': 1e6, 'R3': 25e3, 'R4': 56e3,
        'C1': 500e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    JTM45 = {  # JTM 45
        'R1': 250e3, 'R2': 1e6, 'R3': 25e3, 'R4': 33e3,
        'C1': 270e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    MLEAD = {  # 67 Major Lead 200
        'R1': 250e3, 'R2': 1e6, 'R3': 25e3, 'R4': 33e3,
        'C1': 500e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    M2199 = {  # M2199 30W solid state
        'R1': 250e3, 'R2': 250e3, 'R3': 25e3, 'R4': 56e3,
        'C1': 250e-12, 'C2': 47e-9, 'C3': 47e-9
    }
    
    # Vox Models
    AC30 = {  # 59/86 AC-30
        'R1': 1e6, 'R2': 1e6, 'R3': 10e3, 'R4': 100e3,
        'C1': 50e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    AC15 = {  # VOX AC-15
        'R1': 220e3, 'R2': 220e3, 'R3': 220e3, 'R4': 100e3,
        'C1': 470e-12, 'C2': 100e-9, 'C3': 47e-9
    }
    
    # Other Manufacturers
    SOLDANO = {  # Soldano SLO 100
        'R1': 250e3, 'R2': 1e6, 'R3': 25e3, 'R4': 47e3,
        'C1': 470e-12, 'C2': 20e-9, 'C3': 20e-9
    }
    SOVTEK = {  # MIG 100 H
        'R1': 500e3, 'R2': 1e6, 'R3': 10e3, 'R4': 47e3,
        'C1': 470e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    PEAVEY = {  # c20
        'R1': 250e3, 'R2': 250e3, 'R3': 20e3, 'R4': 68e3,
        'C1': 270e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    IBANEZ = {  # gx20
        'R1': 250e3, 'R2': 250e3, 'R3': 10e3, 'R4': 100e3,
        'C1': 270e-12, 'C2': 100e-9, 'C3': 40e-9
    }
    ROLAND = {  # Cube 60
        'R1': 250e3, 'R2': 250e3, 'R3': 10e3, 'R4': 41e3,
        'C1': 240e-12, 'C2': 33e-9, 'C3': 82e-9
    }
    AMPEG = {  # VL 501
        'R1': 250e3, 'R2': 1e6, 'R3': 25e3, 'R4': 32e3,
        'C1': 470e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    AMPEG_REV = {  # reverbrocket
        'R1': 250e3, 'R2': 250e3, 'R3': 10e3, 'R4': 100e3,
        'C1': 100e-12, 'C2': 100e-9, 'C3': 47e-9
    }
    BOGNER = {  # Triple Giant Preamp
        'R1': 250e3, 'R2': 1e6, 'R3': 33e3, 'R4': 51e3,
        'C1': 220e-12, 'C2': 15e-9, 'C3': 47e-9
    }
    GROOVE = {  # Trio Preamp
        'R1': 220e3, 'R2': 1e6, 'R3': 22e3, 'R4': 68e3,
        'C1': 470e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    CRUNCH = {  # Hughes&Kettner
        'R1': 220e3, 'R2': 220e3, 'R3': 10e3, 'R4': 100e3,
        'C1': 220e-12, 'C2': 47e-9, 'C3': 47e-9
    }
    FENDER_BLUES = {  # Fender Blues Junior
        'R1': 250e3, 'R2': 250e3, 'R3': 25e3, 'R4': 100e3,
        'C1': 250e-12, 'C2': 22e-9, 'C3': 22e-9
    }
    FENDER_DEFAULT = {  # Fender Default
        'R1': 250e3, 'R2': 250e3, 'R3': 10e3, 'R4': 100e3,
        'C1': 250e-12, 'C2': 100e-9, 'C3': 47e-9
    }
    FENDER_DEVILLE = {  # Fender Hot Rod Deville
        'R1': 250e3, 'R2': 250e3, 'R3': 25e3, 'R4': 130e3,
        'C1': 250e-12, 'C2': 100e-9, 'C3': 22e-9
    }
    GIBSON = {  # gs12 reverbrocket
        'R1': 1e6, 'R2': 1e6, 'R3': 94e3, 'R4': 270e3,
        'C1': 25e-12, 'C2': 60e-9, 'C3': 20e-9
    }
    ENGL = {  # engl
        'R1': 250e3, 'R2': 1e6, 'R3': 20e3, 'R4': 100e3,
        'C1': 600e-12, 'C2': 47e-9, 'C3': 47e-9
    }
    
class Tonestack(ProcessorsBase):
    """
    A differentiable implementation of guitar amplifier tonestack.
    
    Args:
        sample_rate (int): Sampling rate in Hz
        preset (str): Amplifier model preset to use (default: 'bassman')
    """
    def __init__(self, sample_rate=44100, preset='bassman'):
        super().__init__(sample_rate)
        self.filter = IIRFilter(order=3, backend='fsm')  # Third order filter
        self.preset = TonestackPreset[preset.upper()].value
        
    def _register_default_parameters(self):
        """Set default parameter ranges"""
        self.params = {
            'bass': EffectParam(min_val=0.0, max_val=1.0),
            'mid': EffectParam(min_val=0.0, max_val=1.0),
            'treble': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def _calculate_coefficients(self, t: torch.Tensor, m: torch.Tensor, l: torch.Tensor) -> tuple:
        """
        Calculate filter coefficients based on control values and component values.
        
        Args:
            t (torch.Tensor): Treble control (0-1)
            m (torch.Tensor): Middle control (0-1)
            l (torch.Tensor): Bass control (0-1)
            
        Returns:
            tuple: (numerator coefficients, denominator coefficients)
        """
        # Get component values
        R1, R2, R3, R4 = [self.preset[k] for k in ['R1', 'R2', 'R3', 'R4']]
        C1, C2, C3 = [self.preset[k] for k in ['C1', 'C2', 'C3']]
        
        # Convert bass control for better response
        l = torch.exp((l - 1) * 3.4)
        
        # Calculate analog coefficients
        b1 = (t*C1*R1 + m*C3*R3 + l*(C1*R2 + C2*R2) + (C1*R3 + C2*R3))
        
        b2 = (t*(C1*C2*R1*R4 + C1*C3*R1*R4) - 
              m*m*(C1*C3*R3*R3 + C2*C3*R3*R3) +
              m*(C1*C3*R1*R3 + C1*C3*R3*R3 + C2*C3*R3*R3) +
              l*(C1*C2*R1*R2 + C1*C2*R2*R4 + C1*C3*R2*R4) +
              l*m*(C1*C3*R2*R3 + C2*C3*R2*R3) +
              (C1*C2*R1*R3 + C1*C2*R3*R4 + C1*C3*R3*R4))
              
        b3 = (l*m*(C1*C2*C3*R1*R2*R3 + C1*C2*C3*R2*R3*R4) -
              m*m*(C1*C2*C3*R1*R3*R3 + C1*C2*C3*R3*R3*R4) +
              m*(C1*C2*C3*R1*R3*R3 + C1*C2*C3*R3*R3*R4) +
              t*C1*C2*C3*R1*R3*R4 - t*m*C1*C2*C3*R1*R3*R4 +
              t*l*C1*C2*C3*R1*R2*R4)
              
        a0 = torch.ones_like(t)
        
        a1 = ((C1*R1 + C1*R3 + C2*R3 + C2*R4 + C3*R4) +
              m*C3*R3 + l*(C1*R2 + C2*R2))
              
        a2 = (m*(C1*C3*R1*R3 - C2*C3*R3*R4 + C1*C3*R3*R3 + C2*C3*R3*R3) +
              l*m*(C1*C3*R2*R3 + C2*C3*R2*R3) -
              m*m*(C1*C3*R3*R3 + C2*C3*R3*R3) +
              l*(C1*C2*R2*R4 + C1*C2*R1*R2 + C1*C3*R2*R4 + C2*C3*R2*R4) +
              (C1*C2*R1*R4 + C1*C3*R1*R4 + C1*C2*R3*R4 + C1*C2*R1*R3 + 
               C1*C3*R3*R4 + C2*C3*R3*R4))
               
        a3 = (l*m*(C1*C2*C3*R1*R2*R3 + C1*C2*C3*R2*R3*R4) -
              m*m*(C1*C2*C3*R1*R3*R3 + C1*C2*C3*R3*R3*R4) +
              m*(C1*C2*C3*R3*R3*R4 + C1*C2*C3*R1*R3*R3 - C1*C2*C3*R1*R3*R4) +
              l*C1*C2*C3*R1*R2*R4 +
              C1*C2*C3*R1*R3*R4)
        
        # Convert to digital domain using bilinear transform
        c = 2 * self.sample_rate
        
        B0 = -b1*c - b2*c**2 - b3*c**3
        B1 = -b1*c + b2*c**2 + 3*b3*c**3
        B2 = b1*c + b2*c**2 - 3*b3*c**3
        B3 = b1*c - b2*c**2 + b3*c**3
        
        A0 = -a0 - a1*c - a2*c**2 - a3*c**3
        A1 = -3*a0 - a1*c + a2*c**2 + 3*a3*c**3
        A2 = -3*a0 + a1*c + a2*c**2 - 3*a3*c**3
        A3 = -a0 + a1*c - a2*c**2 + a3*c**3
        
        # Stack coefficients for IIR filter
        Bs = torch.stack([B0, B1, B2, B3], dim=-1)
        As = torch.stack([A0, A1, A2, A3], dim=-1)
        
        return Bs, As
        
    def process(self, x: torch.Tensor, norm_params: Dict[str, torch.Tensor], 
                dsp_params: Union[Dict[str, torch.Tensor], None] = None) -> torch.Tensor:
        """
        Process input signal through the tonestack.
        
        Args:
            x (torch.Tensor): Input signal [B x C x T]
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0-1)
            dsp_params (Dict[str, torch.Tensor], optional): Direct DSP parameters
            
        Returns:
            torch.Tensor: Processed signal [B x C x T]
        """
        check_params(norm_params, dsp_params)
        
        if norm_params is not None:
            params = self.map_parameters(norm_params)
        else:
            params = dsp_params
            
        # Get control values
        t = params['treble']
        m = params['mid']
        l = params['bass']
        
        # Calculate filter coefficients
        Bs, As = self._calculate_coefficients(t, m, l)
        
        # Apply IIR filter
        y = self.filter(x, Bs, As)
        
        return y

