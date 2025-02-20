Distortion
=================

This section covers several types of distortion including various waveshaping functions and bit-crushing effects.

TanHDist
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.TanHDist
   :members:
   :undoc-members:
   :show-inheritance:

   A smooth distortion effect using the hyperbolic tangent (tanh) function for waveshaping.
   Provides a musical, analog-style saturation with gradual onset and natural compression
   characteristics. The drive parameter controls the intensity of the distortion.

SoftDist
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.SoftDist
   :members:
   :undoc-members:
   :show-inheritance:

   A soft-clipping distortion that provides a smooth transition between clean and distorted signals.
   Uses a polynomial transfer function to create warm overdrive characteristics similar to
   tube amplifier saturation. Features drive and tone controls for versatile sound shaping.

HardDist
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.HardDist
   :members:
   :undoc-members:
   :show-inheritance:

   A hard-clipping distortion that abruptly limits signal peaks above a threshold, creating
   aggressive distortion with rich harmonic content. Useful for creating intense distortion
   effects typical of high-gain guitar amplifiers and fuzz pedals.

DoubleSoftDist
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.DoubleSoftDist
   :members:
   :undoc-members:
   :show-inheritance:

   A dual-stage soft clipping distortion that cascades two soft-clipping stages for more
   complex harmonic generation. Provides additional harmonic richness while maintaining
   musical qualities of soft-clipping. Features independent drive controls for each stage.

CubicDist
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.CubicDist
   :members:
   :undoc-members:
   :show-inheritance:

   A distortion effect based on a cubic polynomial transfer function. Creates asymmetric
   clipping characteristics that add even and odd harmonics to the signal. Provides a
   distinctive sound quality useful for bass and guitar processing.

RectifierDist
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.RectifierDist
   :members:
   :undoc-members:
   :show-inheritance:

   A distortion effect that implements half or full-wave rectification, similar to diode
   clipping circuits. Creates characteristic asymmetric distortion with rich harmonic
   content, particularly useful for aggressive sound design.

BitCrusher
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.BitCrusher
   :members:
   :undoc-members:
   :show-inheritance:

   A digital distortion effect that reduces the bit depth and sample rate of the input
   signal, creating characteristic "lo-fi" artifacts. Features controls for bit depth
   reduction and sample rate decimation, enabling various retro digital effects.

ExponentialDist
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.ExponentialDist
   :members:
   :undoc-members:
   :show-inheritance:

   A distortion effect using exponential transfer functions to create extreme saturation
   characteristics. Provides unique timbral shaping with aggressive harmonic generation,
   suitable for special effects and sound design.

ArcTanDist
------------------

.. autoclass:: diffFx_pytorch.processors.distortion.ArcTanDist
   :members:
   :undoc-members:
   :show-inheritance:

   A smooth distortion effect using the arctangent function for waveshaping. Similar to
   tanh distortion but with slightly different harmonic characteristics. Provides musical
   saturation with natural compression and harmonically rich overtones.