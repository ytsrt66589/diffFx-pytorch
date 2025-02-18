Filters
=================

This section covers several types of filters including fir filter, biquad filter, and linkwitzriley filter.

FIRFilter
------------------

.. autoclass:: diffFx_pytorch.processors.filters.FIRFilter
   :members:
   :undoc-members:
   :show-inheritance:

   A finite impulse response (FIR) filter implementation that processes signals using a moving weighted sum 
   of input samples. This type of filter offers linear phase response and inherent stability. The filter's response 
   is determined by its coefficients.
    
BiquadFilter
------------------

.. autoclass:: diffFx_pytorch.processors.filters.BiquadFilter
   :members:
   :undoc-members:
   :show-inheritance:

   A second-order infinite impulse response (IIR) filter implementation, also known as a biquad filter, 
   that provides efficient filtering with configurable frequency response characteristics. It supports 
   multiple filter types including lowpass, highpass, bandpass, notch, peaking, and shelving filters. 
   Each filter type is controlled through parameters such as frequency, Q factor, and gain, allowing 
   flexible frequency response shaping with minimal computational cost.

LinkwitzRileyFilter
--------------------

.. autoclass:: diffFx_pytorch.processors.filters.LinkwitzRileyFilter
   :members:
   :undoc-members:
   :show-inheritance:

   A specialized crossover filter implementation that provides precise frequency band splitting with 
   optimal phase and magnitude response characteristics. It uses a cascaded design of two Butterworth 
   filters to achieve -24 dB/octave slopes with flat magnitude response at the crossover point and 
   zero phase difference between outputs. This makes it particularly suitable for audio crossover 
   networks and multi-band processing applications.
