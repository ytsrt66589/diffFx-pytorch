# diffFx-pytorch

A PyTorch-based library for differentiable audio effects processing, enabling deep learning integration with professional audio processing algorithms.

## Overview

diffFx-pytorch provides a collection of differentiable audio effects processors that can be seamlessly integrated into neural network architectures. The library implements common audio processing algorithms with PyTorch, making them end-to-end differentiable while maintaining professional audio quality.


## Installation

```bash
pip install diffFx-pytorch
```

## Quick Start

```python
import torch
from difffx.processors.dynamics import Compressor

# Create a compressor
compressor = Compressor(sample_rate=44100)

# Process audio with direct DSP parameters
output = compressor(input_audio, dsp_params={
    'threshold_db': -20.0,
    'ratio': 4.0,
    'knee_db': 6.0,
    'attack_ms': 5.0,
    'release_ms': 50.0,
    'makeup_db': 0.0
})
```

## Neural Network Integration

The library supports deep learning integration through normalized parameters:

```python
import torch
import torch.nn as nn
from difffx.processors import MultiBandCompressor

# Create a neural network controller
class CompressorNet(nn.Module):
    def __init__(self, input_size, num_params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_params),
            nn.Sigmoid()  # Output in range [0,1]
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize processor and network
mb_comp = MultiBandCompressor(sample_rate=44100, num_bands=3)
num_params = mb_comp.count_num_parameters()
controller = CompressorNet(input_size=16, num_params=num_params)

# Process audio with predicted parameters
features = torch.randn(batch_size, 16)
norm_params = controller(features)
output = mb_comp(input_audio, norm_params=norm_params)
```

## Features

### Implemented Effects ✓🚀
- **EQ**
  - [x] ToneStack
  - [x] Graphic Equalizer
  - [x] Parametric Equalizer
  - [] Dynamic EQ
- **Dynamics**
  - [x] Compressor 
  - [x] Multi-band Compressor
  - [x] Limiter
  - [x] Multi-band Limiter
  - [] Expander
  - [] Multi-band Expander
  - [] Noise Gate
  - [] Multi-band Noise Gate
  - [] Transient Shaper 
  - [] Multi-band Transient Shaper
- **Delay**
  - [x] Basic Delay
  - [x] Feedback Basic Delay
  - [x] Slapback Delay
  - [x] Ping-pong Delay
  - [x] Multi-taps Delay
- **Spatial**
  - [x] Stereo Panning
  - [x] Stereo Widener
  - [x] Stereo Imager
  - [x] Stereo Enhancer (Haas Effect)
- **Modulation**
  - [] Chorus
  - [] Multi-voice Chorus
  - [] Stereo Chorus
  - [] Flanger
  - [] Feedback Flanger
  - [] Stereo Flanger 
  - [] Phaser 
  - [] AutoWah 
  - [] Tremelo 
  - [] Ring Modulation
- **Reverb**
  - [] ConvIR Reverb
  - [] Noise Shape Reverb
  - [] Feedback Delay Network (FDN)
- **Distortion (Nonlinear)**
  - [] TanH
  - [] Hard/Soft/Double Soft Clipper
  - [] Bit Crusher 
  - [] Exciter 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Check the to-do list above for effects that haven't been implemented yet.

## Citation

If you use diffFx-pytorch in your research, please cite:

```bibtex
@software{difffx_pytorch,
  title = {diffFx-pytorch: Differentiable Audio Effects Processing in PyTorch},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/difffx-pytorch}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
