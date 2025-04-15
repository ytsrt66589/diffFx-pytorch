Welcome to diffFx-pytorch
========================

A PyTorch-based library for differentiable audio effects processing, enabling deep learning integration with professional audio processing algorithms.

Disclaimer
----------

This is my personal practice project to understand audio effect processors and is designed for deep learning frameworks. Several excellent libraries already exist, such as `GRAFX <https://github.com/sh-lee97/grafx>`_, `dasp-pytorch <https://github.com/csteinmetz1/dasp-pytorch>`_, `NablAFx <https://arxiv.org/abs/2502.11668>`_, and `torchcomp <https://github.com/DiffAPF/torchcomp>`_. Some of my code is inspired by these libraries, and I'm grateful to their developers for implementing several fundamental processors. My core extension will be developing human-interpretable effect processors, where the parameters of each processor can be easily understood by humans. Current code is not fully tested yet, be careful to use!

Overview
--------

**diffFx-pytorch** provides a collection of differentiable audio effects processors that can be integrated into neural network architectures. The library implements common audio processing algorithms with PyTorch, making them end-to-end differentiable while maintaining professional audio quality.

Why diffFx-pytorch
-----------------

* End-to-end differentiable implementation enabling gradient flow through all processing stages

* Human-interpretable parameters matching industry standards (e.g., threshold in dB, ratio in compression ratio) for intuitive control

* Built on native PyTorch operations for seamless deep learning integration and GPU acceleration

* Professional-grade audio quality suitable for production use

Features
--------

Implemented Effects 🎛️
~~~~~~~~~~~~~~~~~~~~~

* **Utilities**
    * Send
    * Mid/Side Processing

* **Linear Gain**
    * Gain
    * Fade in/out (coming soon)

* **EQ**
    * ToneStack
    * Graphic Equalizer
    * Parametric Equalizer
    * Dynamic EQ (coming soon)

* **Dynamics**
    * Compressor
    * Multi-band Compressor
    * Limiter
    * Multi-band Limiter
    * Expander
    * Multi-band Expander
    * Noise Gate
    * Multi-band Noise Gate
    * Transient Shaper (coming soon)
    * Multi-band Transient Shaper (coming soon)

* **Delay**
    * Basic Delay
    * Feedback Basic Delay
    * Slapback Delay
    * Ping-pong Delay
    * Multi-taps Delay

* **Spatial**
    * Stereo Panning
    * Stereo Widener
    * Stereo Imager (Multi-band Widener)
    * Stereo Enhancer (Haas Effect)

* **Modulation**
    * Chorus
    * Multi-voice Chorus
    * Stereo Chorus
    * Flanger
    * Feedback Flanger
    * Stereo Flanger
    * Phaser
    * AutoWah (coming soon)
    * Tremelo (coming soon)
    * Ring Modulation (coming soon)

* **Reverb**
    * ConvIR Reverb (coming soon)
    * Noise Shape Reverb (coming soon)
    * Feedback Delay Network (FDN) (coming soon)

* **Distortion (Nonlinear)**
    * TanH
    * Hard/Soft/Double-Soft/Cubic/ArcTanh/Rectifier/Exponential Clipper
    * Bit Crusher

Installation
-----------

.. code-block:: bash

    pip install diffFx-pytorch

or

.. code-block:: bash

    git clone https://github.com/ytsrt66589/diffFx-pytorch.git
    cd diffFx-pytorch
    pip install -e .

Quick Start
----------

Basic Usage
~~~~~~~~~~

.. code-block:: python

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

Parameter Range Customization
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize the parameter ranges of each processor:

.. code-block:: python

    from diffFx_pytorch.processors.dynamics import Compressor 
    from diffFx_pytorch.processors import EffectParam

    # Create a compressor with custom parameter ranges
    compressor = Compressor(
        sample_rate=44100,
        param_range={
            'threshold_db': EffectParam(min_val=-60.0, max_val=0.0),
            'ratio': EffectParam(min_val=1.0, max_val=20.0),
            'knee_db': EffectParam(min_val=0.0, max_val=12.0),
            'attack_ms': EffectParam(min_val=0.1, max_val=100.0),
            'release_ms': EffectParam(min_val=10.0, max_val=1000.0),
            'makeup_db': EffectParam(min_val=-12.0, max_val=12.0)
        }
    )

Neural Network Integration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Parameter Conversion
~~~~~~~~~~~~~~~~~~

Each processor provides methods to convert between DSP and normalized parameters:

.. code-block:: python

    # Convert DSP parameters to normalized
    norm_params = compressor.demap_parameters(dsp_params)
    
    # Convert normalized parameters to DSP
    dsp_params = compressor.map_parameters(norm_params)

Examples
--------

Understanding the sound characteristic of each processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check `examples/processors/notebook <examples/processors/notebook>`_ to see how each processor affect sound.

Citation
--------

If you use diffFx-pytorch in your research, please cite:

.. code-block:: bibtex

    @software{difffx_pytorch,
        title = {diffFx-pytorch: Differentiable Audio Effects Processing in PyTorch},
        author = {Yen-Tung Yeh},
        year = {2025},
        url = {https://github.com/ytsrt66589/difffx-pytorch}
    }

License
-------

This project is licensed under the Apache License - see the `LICENSE <LICENSE>`_ file for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/installation
   usage/quickstart
   api/index