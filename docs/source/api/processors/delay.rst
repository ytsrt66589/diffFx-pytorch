Delay
=================

This section covers several types of delay including basic delay, feedback delay, slapback delay, ping-pong delay and multitaps delay.

BasicDelay
------------------

.. autoclass:: diffFx_pytorch.processors.delay.BasicDelay
   :members:
   :undoc-members:
   :show-inheritance:

   A simple delay line that creates a single echo of the input signal after a specified time delay. 
   It provides control over the delay time and mix ratio between the dry (original) and wet (delayed) 
   signals, offering basic time-based effects without feedback.
    
BasicFeedbackDelay
------------------

.. autoclass:: diffFx_pytorch.processors.delay.BasicFeedbackDelay
   :members:
   :undoc-members:
   :show-inheritance:

   A delay effect that includes a feedback path, allowing the delayed signal to be fed back into 
   the input. This creates multiple, gradually decaying echoes. Features controls for delay time, 
   feedback amount, and wet/dry mix, enabling creation of everything from subtle space to rhythmic 
   echo patterns.

SlapbackDelay
------------------

.. autoclass:: diffFx_pytorch.processors.delay.SlapbackDelay
   :members:
   :undoc-members:
   :show-inheritance:

   A specialized short delay effect that emulates the distinctive "doubling" sound of vintage tape 
   delays. Uses very short delay times (typically 60-120ms) with minimal to no feedback, creating 
   a tight, distinctive echo that was popular in early rock and roll recordings.

PingPongDelay
------------------

.. autoclass:: diffFx_pytorch.processors.delay.PingPongDelay
   :members:
   :undoc-members:
   :show-inheritance:

   A stereo delay effect where the echoes alternate between left and right channels, creating 
   a "ping-pong" effect across the stereo field. Each echo bounces from one channel to the other 
   while decreasing in amplitude, producing a wide spatial effect with rhythmic possibilities.

MultiTapsDelay
------------------

.. autoclass:: diffFx_pytorch.processors.delay.MultiTapsDelay
   :members:
   :undoc-members:
   :show-inheritance:

   A complex delay effect that creates multiple delayed copies (taps) of the input signal, each 
   with independent timing, level, and panning controls. This allows for creation of complex 
   rhythmic patterns and spatial effects by precisely controlling the timing and placement of 
   each echo.
