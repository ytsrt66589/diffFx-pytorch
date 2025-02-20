Modulation Effects
==================

This section covers various modulation effects including chorus, flanger, and phaser variations.

Chorus
-------

.. autoclass:: diffFx_pytorch.processors.modulation.Chorus
   :members:
   :undoc-members:
   :show-inheritance:

   A chorus effect that creates a shimmering, thickening sound by adding slightly detuned 
   and delayed copies of the original signal. This processor adds depth and width to the 
   original sound by simulating multiple slightly out-of-tune instruments playing together.

MultiVoiceChorus
-----------------

.. autoclass:: diffFx_pytorch.processors.modulation.MultiVoiceChorus
   :members:
   :undoc-members:
   :show-inheritance:

   An advanced chorus effect that generates multiple detuned and delayed voices from the 
   original signal. This effect provides more complex and rich spatial modulation compared 
   to traditional single-voice chorus effects, allowing for more intricate sound design 
   and spatial enhancement.

StereoChorus
-------------

.. autoclass:: diffFx_pytorch.processors.modulation.StereoChorus
   :members:
   :undoc-members:
   :show-inheritance:

   A stereo-specific chorus effect that creates a wide, immersive modulation by applying 
   independent chorus processing to left and right channels. This approach enhances the 
   stereo image and provides a more expansive sound stage compared to mono chorus effects.

Flanger
--------

.. autoclass:: diffFx_pytorch.processors.modulation.Flanger
   :members:
   :undoc-members:
   :show-inheritance:

   A classic modulation effect that creates a sweeping, metallic sound by mixing the 
   original signal with a slightly delayed, time-modulated copy of itself. The effect 
   produces a characteristic whooshing or airplane-like sound through controlled 
   signal delay and feedback.

StereoFlanger
--------------

.. autoclass:: diffFx_pytorch.processors.modulation.StereoFlanger
   :members:
   :undoc-members:
   :show-inheritance:

   A stereo implementation of the flanger effect that applies independent flanger 
   processing to left and right channels. This approach creates a more complex and 
   spatially interesting modulation effect compared to mono flanger implementations.

FeedbackFlanger
----------------

.. autoclass:: diffFx_pytorch.processors.modulation.FeedbackFlanger
   :members:
   :undoc-members:
   :show-inheritance:

   An extended flanger effect that incorporates feedback to create more pronounced 
   and resonant modulation. The feedback mechanism allows for more intense and 
   distinctive sweeping sounds, adding additional harmonic complexity to the 
   original signal.

Phaser
-------

.. autoclass:: diffFx_pytorch.processors.modulation.Phaser
   :members:
   :undoc-members:
   :show-inheritance:

   A phase-shifting modulation effect that creates a sweeping, swirling sound by 
   splitting the signal into multiple all-pass filtered paths and recombining them. 
   This results in a distinctive, swooping modulation that adds movement and depth 
   to the original sound.