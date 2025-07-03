MS Processing & Send
=================

This section covers mid/side processing and send effects.

MSProcessing
------------------

.. autoclass:: diffFx_pytorch.processors.utilities.MidSideProc
   :members:
   :undoc-members:
   :show-inheritance:

   A processor that splits the input signal into mid and side components and applies separate processing to each.

Send
----

.. autoclass:: diffFx_pytorch.processors.utilities.SendProc
   :members:
   :undoc-members:
   :show-inheritance:

   A processor that sends the input signal to a separate processing chain and mixes the output with the original signal.