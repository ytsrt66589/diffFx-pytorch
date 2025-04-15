Dynamics Processors
=================

This section covers dynamics processors including compressors, limiters, and their multi-band variants.

Compressor
----------

.. autoclass:: diffFx_pytorch.processors.dynamics.Compressor
   :members:
   :undoc-members:
   :show-inheritance:

   A feedforward dynamic range compressor with three knee types (hard, quadratic, exponential) 
   and two smoothing options (ballistics, IIR).

Multi-band Compressor
------------------

.. autoclass:: diffFx_pytorch.processors.dynamics.MultiBandCompressor
   :members:
   :undoc-members:
   :show-inheritance:

   A multi-band compressor that splits the input into frequency bands and applies 
   independent compression to each band.

Limiter
-------

.. autoclass:: diffFx_pytorch.processors.dynamics.Limiter
   :members:
   :undoc-members:
   :show-inheritance:

   A specialized compressor configured for limiting with fast attack times and high ratios.

Multi-band Limiter
---------------

.. autoclass:: diffFx_pytorch.processors.dynamics.MultiBandLimiter
   :members:
   :undoc-members:
   :show-inheritance:

   A multi-band limiter that applies independent limiting to different frequency bands.

Expander 
--------

.. autoclass:: diffFx_pytorch.processors.dynamics.Expander
   :members:
   :undoc-members:
   :show-inheritance:   

   A feedforward expander with three knee types (hard, quadratic, exponential) 
   and two smoothing options (ballistics, IIR). 

Multi-band Expander
-------------------

.. autoclass:: diffFx_pytorch.processors.dynamics.MultiBandExpander
   :members:
   :undoc-members:
   :show-inheritance:  

   A multi-band expander that applies independent expansion to different frequency bands.

Noise Gate
----------

.. autoclass:: diffFx_pytorch.processors.dynamics.NoiseGate
   :members:
   :undoc-members:
   :show-inheritance:   

   A noise gate that reduces the volume of quiet audio signals.   

Multi-band Noise Gate
---------------------

.. autoclass:: diffFx_pytorch.processors.dynamics.MultiBandNoiseGate
   :members:
   :undoc-members:
   :show-inheritance:   

   A multi-band noise gate that applies independent noise gating to different frequency bands.

Transient Shaper
---------------

.. autoclass:: diffFx_pytorch.processors.dynamics.TransientShaper
   :members:
   :undoc-members:
   :show-inheritance:

   A transient shaper that can alter the transient characteristics