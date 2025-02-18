Equalizer
=================

This section covers several types of equalizers including tonestack, graphic equalizers, and parametric equalizers.

Tonestack
------------------

.. autoclass:: diffFx_pytorch.processors.eq.Tonestack
   :members:
   :undoc-members:
   :show-inheritance:

   A classic three-band equalizer that emulates the behavior of analog tone controls found in guitar amplifiers. 
   Typically includes bass, middle, and treble controls that interact with each other similar to passive analog circuits.
    
GraphicEqualizer
------------------

.. autoclass:: diffFx_pytorch.processors.eq.GraphicEqualizer
   :members:
   :undoc-members:
   :show-inheritance:

   A multi-band equalizer that divides the frequency spectrum into fixed bands (typically octaves or third-octaves), 
   where each band can be boosted or cut independently using slider controls. The center frequencies and bandwidths 
   are fixed, and the gains can be adjusted to create a visual representation of the frequency response.

ParametricEqualizer
--------------------

.. autoclass:: diffFx_pytorch.processors.eq.ParametricEqualizer
   :members:
   :undoc-members:
   :show-inheritance:

   A flexible equalizer that allows precise control over specific frequency ranges through adjustable parameters 
   including center frequency, bandwidth (Q factor), and gain for each band. Unlike graphic EQs, the center 
   frequencies and bandwidths can be freely adjusted, typically offering more surgical sound shaping capabilities.
