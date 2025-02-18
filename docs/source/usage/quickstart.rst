Quick Start Guide
===============

Basic Usage
----------

Here's a simple example using the compressor:

.. code-block:: python

   import torch
   from diffFx_pytorch.processors.dynamics import Compressor

   # Create a compressor
   compressor = Compressor(sample_rate=44100)

   # Process audio
   output = compressor(input_audio, {
       'threshold_db': -20.0,
       'ratio': 4.0,
       'attack_ms': 5.0,
       'release_ms': 50.0
   })