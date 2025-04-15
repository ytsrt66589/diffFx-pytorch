# diffFx-pytorch

A PyTorch-based library for differentiable audio effects processing, enabling deep learning integration with professional audio processing algorithms.


Disclaimer: This is my personal practice project to understand audio effect processors and is designed for deep learning frameworks. Several excellent libraries already exist, such as [GRAFX](https://github.com/sh-lee97/grafx), [dasp-pytorch](https://github.com/csteinmetz1/dasp-pytorch), [NablAFx](https://arxiv.org/abs/2502.11668), and [torchcomp](https://github.com/DiffAPF/torchcomp). Some of my code is inspired by these libraries, and I'm grateful to their developers for implementing several fundamental processors. My core extension will be developing human-interpretable effect processors, where the parameters of each processor can be easily understood by humans. Current code is not fully test yet, be careful to use !!!!!!!!!!! 

[Doc](https://difffx-pytorch.readthedocs.io/en/latest/)
## Overview

**diffFx-pytorch** provides a collection of differentiable audio effects processors that can be  integrated into neural network architectures. The library implements common audio processing algorithms with PyTorch, making them end-to-end differentiable while maintaining professional audio quality.


## Installation

(Haven't upload to pypi, so you can only try install via this repo)
```bash
pip install diffFx-pytorch
```

or 

```bash
git clone https://github.com/ytsrt66589/diffFx-pytorch.git
cd diffFx-pytorch
pip install -e .
```

## Quick Start

```python
import torch
from diffFx_pytorch.processors.dynamics import Compressor

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
from diffFx_pytorch.processors.dynamics import MultiBandCompressor

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

## Examples 

### Understanding the sound characteristic of each processor
Check [examples/processors/notebook](examples/processors/notebook) to see how each processor affect sound. 


## Features
### Implemented Effects 🎛️ 
- **Utilities**
  - [x] Send 
  - [x] Mid/Side Processing 
- **Linear Gain**
  - [x] Gain 
  - [] Fade in/out
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
  - [x] Expander
  - [x] Multi-band Expander
  - [x] Noise Gate
  - [x] Multi-band Noise Gate
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
  - [x] Stereo Imager (Multi-band Widener)
  - [x] Stereo Enhancer (Haas Effect)
- **Modulation**
  - [x] Chorus
  - [x] Multi-voice Chorus
  - [x] Stereo Chorus
  - [x] Flanger
  - [x] Feedback Flanger
  - [x] Stereo Flanger 
  - [x] Phaser 
  - [] AutoWah 
  - [] Tremelo 
  - [] Ring Modulation
- **Reverb**
  - [] ConvIR Reverb
  - [] Noise Shape Reverb
  - [] Feedback Delay Network (FDN)
- **Distortion (Nonlinear)**
  - [x] TanH
  - [x] Hard/Soft/Double-Soft/Cubic/ArcTanh/Rectifier/Exponential Clipper
  - [x] Bit Crusher 



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Check the to-do list above for effects that haven't been implemented yet.

## Citation

If you use diffFx-pytorch in your research, please cite:

```bibtex
@software{difffx_pytorch,
  title = {diffFx-pytorch: Differentiable Audio Effects Processing in PyTorch},
  author = {Yen-Tung Yeh},
  year = {2024},
  url = {https://github.com/ytsrt66589/difffx-pytorch}
}
```

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
