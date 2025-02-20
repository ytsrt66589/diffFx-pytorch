Welcome to diffFx-pytorch
========================

A comprehensive PyTorch-based library for differentiable audio effects processing, offering a wide range of professional audio processors that seamlessly integrate with deep learning architectures.

Overview
--------

diffFx-pytorch implements a diverse collection of industry-standard audio effects as differentiable processors in PyTorch. The library makes professional audio processing algorithms end-to-end differentiable while maintaining pristine audio quality, enabling novel applications in neural audio processing and automatic music production.

Why diffFx-pytorch
-----------------

* End-to-end differentiable implementation enabling gradient flow through all processing stages

* Human-interpretable parameters matching industry standards (e.g., threshold in dB, ratio in compression ratio) for intuitive control

* Built on native PyTorch operations for seamless deep learning integration and GPU acceleration

* Professional-grade audio quality suitable for production use

Key Features
-----------

Audio Processors
~~~~~~~~~~~~~~~

* **Equalization**
    * ToneStack
    * Graphic Equalizer
    * Parametric Equalizer

* **Dynamics**
    * Compressor
    * Multi-band Compressor
    * Limiter
    * Multi-band Limiter

* **Delay**
    * Basic Delay
    * Feedback Delay
    * Slapback Delay
    * Ping-pong Delay
    * Multi-taps Delay

* **Spatial**
    * Stereo Panning
    * Stereo Widener
    * Stereo Imager
    * Stereo Enhancer

Implementation Features
~~~~~~~~~~~~~~~~~~~~~

* PyTorch-native operations
* GPU acceleration support
* Efficient batch processing
* Automatic differentiation

Applications
-----------

Music Production
~~~~~~~~~~~~~~~

* Intelligent mixing and mastering
* Adaptive audio processing
* Parameter optimization

Research Applications
~~~~~~~~~~~~~~~~~~~

* Neural audio effect modeling
* Style transfer
* Automatic music production

Installation
-----------

.. code-block:: bash

    pip install diffFx-pytorch

Quick Start
----------

.. code-block:: python

    import torch
    from difffx.processors import Compressor

    # Create a compressor
    compressor = Compressor(sample_rate=44100)

    # Process audio with DSP parameters
    output = compressor(input_audio, dsp_params={
        'threshold_db': -20.0,
        'ratio': 4.0,
        'knee_db': 6.0,
        'attack_ms': 5.0,
        'release_ms': 50.0,
        'makeup_db': 0.0
    })

Neural Network Integration
------------------------

.. code-block:: python

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

    # Process with predicted parameters
    features = torch.randn(batch_size, 16)
    norm_params = controller(features)
    output = mb_comp(input_audio, norm_params=norm_params)

Citation
--------

If you use diffFx-pytorch in your research, please cite:

.. code-block:: bibtex

    @software{difffx_pytorch,
        title = {diffFx-pytorch: Differentiable Audio Effects Processing in PyTorch},
        author = {Your Name},
        year = {2024},
        url = {https://github.com/yourusername/difffx-pytorch}
    }

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/installation
   usage/quickstart
   api/index