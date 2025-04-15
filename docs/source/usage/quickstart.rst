Quick Start Guide
===============

Basic Usage
----------

Here's a simple example using the compressor with direct DSP parameters:

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

Utility Processors
----------------

Mid/Side Processing
~~~~~~~~~~~~~~~~~

The MidSideProc allows you to process the mid (sum) and side (difference) components of a stereo signal separately:

.. code-block:: python

   from diffFx_pytorch.processors.utilities import MidSideProc
   from diffFx_pytorch.processors.dynamics import Compressor

   # Create processors for mid and side channels
   mid_comp = Compressor(sample_rate=44100)
   side_comp = Compressor(sample_rate=44100)
   
   # Create MidSide processor
   ms_processor = MidSideProc(mid_comp)
   
   # Process stereo audio
   stereo_input = torch.randn(batch_size, 2, 44100)  # [batch, channels, samples]
   output = ms_processor(
       stereo_input,
       mult=0.5,
       dsp_mid_params={
           'threshold_db': -20.0,
           'ratio': 4.0,
           'attack_ms': 5.0,
           'release_ms': 50.0
       },
       dsp_side_params={
           'threshold_db': -30.0,
           'ratio': 2.0,
           'attack_ms': 10.0,
           'release_ms': 100.0
       }
   )

Send Processing
~~~~~~~~~~~~~

The SendProc allows you to process a signal in parallel and mix it with the original:

.. code-block:: python

   from diffFx_pytorch.processors.utilities import SendProc
   from diffFx_pytorch.processors.dynamics import Compressor

   # Create a compressor for the send chain
   send_comp = Compressor(sample_rate=44100)
   
   # Create Send processor
   send_processor = SendProc(send_comp)
   
   # Process audio with parallel compression
   input_audio = torch.randn(batch_size, 2, 44100)
   output = send_processor(
       input_audio,
       mult=0.3,  # Mix 30% of the compressed signal
       dsp_params={
           'threshold_db': -20.0,
           'ratio': 4.0,
           'attack_ms': 5.0,
           'release_ms': 50.0
       }
   )

Parameter Range Customization
---------------------------

You can customize the parameter ranges of each processor using the ``param_range`` argument. This is useful when you want to limit the range of parameters for specific applications:

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
------------------------

The library supports deep learning integration through normalized parameters. Here's an example using a multi-band compressor:

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

Parameter Types
-------------

The library supports two types of parameters:

1. DSP Parameters
   ~~~~~~~~~~~~~
   These are the actual audio processing parameters in their natural units (e.g., dB, milliseconds).
   They are passed using the ``dsp_params`` argument.

2. Normalized Parameters
   ~~~~~~~~~~~~~~~~~~~~
   These are parameters normalized to the range [0, 1], suitable for neural network prediction.
   They are passed using the ``norm_params`` argument.

Each processor provides methods to convert between these parameter types:

.. code-block:: python

   # Convert DSP parameters to normalized
   norm_params = compressor.demap_parameters(dsp_params)
   
   # Convert normalized parameters to DSP
   dsp_params = compressor.map_parameters(norm_params)

Complex Effect Chain
------------------

Here's an example of creating a professional mixing chain with neural network control:

.. code-block:: python

    import torch
    import torch.nn as nn
    from diffFx_pytorch.processors.utilities import MidSideProc, SendProc
    from diffFx_pytorch.processors.dynamics import Compressor, MultiBandCompressor
    from diffFx_pytorch.processors.eq import ParametricEQ
    from diffFx_pytorch.processors.spatial import StereoWidener

    class MixingChain(nn.Module):
        def __init__(self, sample_rate=44100):
            super().__init__()
            
            # Create individual processors
            self.eq = ParametricEQ(sample_rate=sample_rate)
            self.comp = Compressor(sample_rate=sample_rate)
            self.mb_comp = MultiBandCompressor(sample_rate=sample_rate, num_bands=3)
            self.widener = StereoWidener(sample_rate=sample_rate)
            
            # Create parallel compression chain
            self.parallel_comp = SendProc(
                Compressor(sample_rate=sample_rate)
            )
            
            # Create mid/side processing chain
            self.mid_side = MidSideProc(
                MultiBandCompressor(sample_rate=sample_rate, num_bands=3)
            )
            
            # Create parameter prediction networks
            self.eq_net = self._create_controller(16, self.eq.count_num_parameters())
            self.comp_net = self._create_controller(16, self.comp.count_num_parameters())
            self.parallel_comp_net = self._create_controller(16, self.comp.count_num_parameters())
            self.mid_net = self._create_controller(16, self.mb_comp.count_num_parameters())
            self.side_net = self._create_controller(16, self.mb_comp.count_num_parameters())
            self.mb_comp_net = self._create_controller(16, self.mb_comp.count_num_parameters())
            self.widener_net = self._create_controller(16, self.widener.count_num_parameters())
        
        def _create_controller(self, input_size, num_params):
            return nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, num_params),
                nn.Sigmoid()  # Output in range [0,1]
            )
        
        def forward(self, x, features):
            # Predict parameters for each processor
            eq_params = self.eq_net(features)
            comp_params = self.comp_net(features)
            parallel_comp_params = self.parallel_comp_net(features)
            mid_params = self.mid_net(features)
            side_params = self.side_net(features)
            mb_comp_params = self.mb_comp_net(features)
            widener_params = self.widener_net(features)
            
            # 1. Initial EQ
            x = self.eq(x, norm_params=eq_params)
            
            # 2. Main compression
            x = self.comp(x, norm_params=comp_params)
            
            # 3. Parallel compression
            x = self.parallel_comp(
                x,
                mult=0.3,  # Mix 30% of parallel compression
                norm_params=parallel_comp_params
            )
            
            # 4. Mid/Side processing
            x = self.mid_side(
                x,
                mult=0.5,
                norm_mid_params=mid_params,
                norm_side_params=side_params
            )
            
            # 5. Multi-band compression
            x = self.mb_comp(x, norm_params=mb_comp_params)
            
            # 6. Stereo widening
            x = self.widener(x, norm_params=widener_params)
            
            return x

    # Create the mixing chain
    mixing_chain = MixingChain(sample_rate=44100)

    # Example usage with features
    batch_size = 4
    features = torch.randn(batch_size, 16)  # Feature vector for parameter prediction
    input_audio = torch.randn(batch_size, 2, 44100)  # [batch, channels, samples]
    output = mixing_chain(input_audio, features)

This example demonstrates:
1. How to create a neural network controller for each processor
2. How to predict parameters from features
3. How to chain multiple processors with neural network control
4. Using both serial and parallel processing with learned parameters
5. Mid/Side processing with separate neural networks for mid and side channels
6. Multi-band processing with learned parameters
7. How to organize feature-based parameter prediction for complex chains