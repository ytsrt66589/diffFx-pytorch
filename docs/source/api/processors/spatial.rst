Spatial Effects
=================

This section covers several types of spatial effects including stereo panning, stereo widening, stereo imager, and stereo enhancer.

StereoPanning
----------------

.. autoclass:: diffFx_pytorch.processors.spatial.StereoPanning
   :members:
   :undoc-members:
   :show-inheritance:

   A stereo positioning processor that controls the balance between left and right channels. 
   It allows for continuous panning across the stereo field, from fully left to fully right, 
   while maintaining consistent overall loudness through amplitude-compensated panning laws.
    
StereoWidener
----------------

.. autoclass:: diffFx_pytorch.processors.spatial.StereoWidener
   :members:
   :undoc-members:
   :show-inheritance:

   A stereo enhancement processor that increases the perceived width of the stereo image 
   by manipulating the mid-side (M/S) representation of the signal. It can expand or 
   contract the stereo field while maintaining mono compatibility and allowing independent 
   control over different frequency ranges.

StereoImager
----------------

.. autoclass:: diffFx_pytorch.processors.spatial.StereoImager
   :members:
   :undoc-members:
   :show-inheritance:

   A multi-band stereo processing tool that provides independent control over the stereo 
   width in different frequency bands. It uses mid-side processing with crossover filters 
   to allow precise adjustment of the stereo image across the frequency spectrum, enabling 
   frequency-dependent stereo manipulation.

StereoEnhancer
----------------

.. autoclass:: diffFx_pytorch.processors.spatial.StereoEnhancer
   :members:
   :undoc-members:
   :show-inheritance:

   A stereo enhancement processor that implements the Haas effect (also known as the precedence effect 
   or law of the first wavefront) to create a wider stereo image. It operates by introducing small 
   time delays (typically 5-35ms) between the left and right channels, exploiting the human auditory 
   system's spatial perception mechanisms. When one channel is delayed relative to the other within 
   this specific time window, the sound appears to come from the direction of the first-arriving sound 
   while maintaining the loudness contribution from both channels, resulting in an enhanced sense of 
   width without phantom center image collapse.
