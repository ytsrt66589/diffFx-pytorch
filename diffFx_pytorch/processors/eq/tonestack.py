import torch 
from typing import Dict, Union
from enum import Enum
from ..base_utils import check_params
from ..base import ProcessorsBase, EffectParam
from ..core.iir import IIRFilter

class TonestackPreset(Enum):
    """Collection of component values modeling various guitar amplifier tonestacks.
    From: https://github.com/mod-audio/guitarix/blob/master/trunk/src/faust/tonestack.dsp?fbclid=IwAR02TRPhiyVm5d_K0Df9KR8gxzbYcZX80NPvzT3ciCMn4r0V-iUJg8yH0YU
    Each preset defines the resistor (R1-R4) and capacitor (C1-C3) values that
    characterize the frequency response of a specific amplifier model. Values are
    provided in standard units (ohms for resistors, farads for capacitors).
    
    Available Models:
        Fender Family:
            - BASSMAN: '59 Bassman 5F6-A
            - MESA: Mesa Boogie Mark
            - TWIN: '69 Twin Reverb AA270
            - PRINCETON: '64 Princeton AA1164
            
        Marshall Family:
            - JCM800: '59/81 JCM-800 Lead 100 2203
            - JCM2000: '81 2000 Lead
            - JTM45: JTM 45
            - MLEAD: '67 Major Lead 200
            
        Vox Family:
            - AC30: '59/86 AC-30
            - AC15: VOX AC-15
            
        Other Manufacturers:
            - SOLDANO: Soldano SLO 100
            - SOVTEK: MIG 100 H
            - PEAVEY: Peavey C20
            - IBANEZ: Ibanez GX20
            - ROLAND: Roland Cube 60
            
    Component Value Format:
        Each preset is a dictionary with the following keys:
            - R1, R2, R3, R4: Resistor values in ohms
            - C1, C2, C3: Capacitor values in farads
            
    Note:
        Component values are based on measurements and modeling of actual
        amplifier circuits. The interaction between these components and
        the control settings creates the characteristic tonal response
        of each amplifier model.
    """
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
    """Differentiable implementation of classic guitar amplifier tonestack circuits.
    
    This processor implements a digital model of analog tonestack circuits found in guitar amplifiers,
    providing control over bass, middle, and treble frequencies. The implementation is based on
    modeling the analog circuit components and their interactions using a third-order IIR filter.

    The transfer function is based on the following analog circuit topology:

    .. math::

        H(s) = \\frac{b_0 + b_1s + b_2s^2 + b_3s^3}{1 + a_1s + a_2s^2 + a_3s^3}

    where coefficients b0-b3 and a1-a3 are functions of:
        - Component values (R1-R4, C1-C3) from the chosen preset
        - Control positions (bass, mid, treble)
        - The coefficients are calculated using circuit analysis equations
        - The bilinear transform is then applied to convert to digital domain
    
    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to 44100.
        preset (str): Amplifier model preset to use. Must be one of the following:
            
            Fender Family:
                - BASSMAN: '59 Bassman 5F6-A
                - MESA: Mesa Boogie Mark
                - TWIN: '69 Twin Reverb AA270
                - PRINCETON: '64 Princeton AA1164
                - FENDER_BLUES: Fender Blues Junior
                - FENDER_DEFAULT: Fender Default
                - FENDER_DEVILLE: Fender Hot Rod Deville
                
            Marshall Family:
                - JCM800: '59/81 JCM-800 Lead 100 2203
                - JCM2000: '81 2000 Lead
                - JTM45: JTM 45
                - MLEAD: '67 Major Lead 200
                - M2199: M2199 30W solid state
                
            Vox Family:
                - AC30: '59/86 AC-30
                - AC15: VOX AC-15
                
            Other Manufacturers:
                - SOLDANO: Soldano SLO 100
                - SOVTEK: MIG 100 H
                - PEAVEY: Peavey C20
                - IBANEZ: Ibanez GX20
                - ROLAND: Roland Cube 60
                - AMPEG: VL 501
                - AMPEG_REV: Ampeg Reverbrocket
                - BOGNER: Triple Giant Preamp
                - GROOVE: Trio Preamp
                - CRUNCH: Hughes&Kettner
                - GIBSON: GS12 Reverbrocket
                - ENGL: Engl

    Parameters Details:
        bass: Low frequency control
            - Range: 0.0 to 1.0
            - Higher values boost low frequencies
        mid: Middle frequency control
            - Range: 0.0 to 1.0
            - Controls presence of middle frequencies
        treble: High frequency control
            - Range: 0.0 to 1.0
            - Higher values boost high frequencies

    Note:
        The processor supports a collection of presets modeling famous guitar amplifier circuits
        including Fender, Marshall, Vox, and other manufacturers. Each preset defines specific
        component values that determine the characteristic sound of that amplifier model.

    Warning:
        When using with neural networks:
            - norm_params must be in range [0, 1]
            - Parameters will be automatically mapped to their ranges
            - Ensure your network output is properly normalized (e.g., using sigmoid)
            - Parameter order must match _register_default_parameters()

    Examples:
        Basic DSP Usage:
            >>> # Create a tonestack with Fender Bassman preset
            >>> tonestack = Tonestack(
            ...     sample_rate=44100,
            ...     preset="bassman"
            ... )
            >>> # Process audio with dsp parameters
            >>> output = tonestack(input_audio, dsp_params={
            ...     'bass': 0.7,
            ...     'mid': 0.5,
            ...     'treble': 0.6
            ... })

        Neural Network Control:
            >>> # 1. Simple parameter prediction
            >>> class TonestackController(nn.Module):
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
            >>> num_params = tonestack.count_num_parameters()  # 3 parameters
            >>> controller = TonestackController(input_size=16, num_params=num_params)
            >>> 
            >>> # Process with features
            >>> features = torch.randn(batch_size, 16)  # Audio features
            >>> norm_params = controller(features)
            >>> output = tonestack(input_audio, norm_params=norm_params)
    """
    def __init__(self, sample_rate=44100, param_range=None, preset='bassman'):
        super().__init__(sample_rate, param_range)
        self.filter = IIRFilter(order=3, backend='fsm')  # Third order filter
        self.preset = TonestackPreset[preset.upper()].value
        
    def _register_default_parameters(self):
        """Register default parameter ranges for the tonestack.

        Sets up the following parameters with their ranges:
            - bass: Low frequency control (0 to 1)
            - mid: Middle frequency control (0 to 1)
            - treble: High frequency control (0 to 1)
            
        Each control maps to a normalized range that interacts with the circuit 
        component values defined by the chosen preset.
        """
        self.params = {
            'bass': EffectParam(min_val=0.0, max_val=1.0),
            'mid': EffectParam(min_val=0.0, max_val=1.0),
            'treble': EffectParam(min_val=0.0, max_val=1.0)
        }
        
    def _calculate_coefficients(self, t: torch.Tensor, m: torch.Tensor, l: torch.Tensor) -> tuple:
        """Calculate filter coefficients based on control values and component values.

        Implements the circuit analysis equations for the tonestack, converting control
        positions and component values into analog filter coefficients, then applying
        the bilinear transform to obtain digital coefficients.
        
        Args:
            t (torch.Tensor): Treble control value (0-1). Shape: (batch,)
            m (torch.Tensor): Middle control value (0-1). Shape: (batch,)
            l (torch.Tensor): Bass control value (0-1). Shape: (batch,)
            
        Returns:
            tuple: (Bs, As) where:
                - Bs: Numerator coefficients for IIR filter. Shape: (batch, 4)
                - As: Denominator coefficients for IIR filter. Shape: (batch, 4)
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
        
    def process(
        self, 
        x: torch.Tensor, 
        norm_params: Union[Dict[str, torch.Tensor], None] = None, 
        dsp_params: Union[Dict[str, Union[float, torch.Tensor]], None] = None
    ) -> torch.Tensor:
        """Process input signal through the tonestack.
        
        Args:
            x (torch.Tensor): Input audio tensor. Shape: (batch, channels, samples)
            norm_params (Dict[str, torch.Tensor]): Normalized parameters (0 to 1)
                Must contain the following keys:
                    - 'bass': Low frequency control (0 to 1)
                    - 'mid': Mid frequency control (0 to 1)
                    - 'treble': High frequency control (0 to 1)
                Each value should be a tensor of shape (batch_size,)
            dsp_params (Dict[str, Union[float, torch.Tensor]], optional): Direct DSP parameters.
                Can specify tone controls as:
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


