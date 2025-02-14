import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


from ..core.fir import FIRConvolution
from ..core.utils import normalize_impulse
from ..core.midside import lr_to_ms, ms_to_lr


class FIRFilter(nn.Module):
    def __init__(self, fir_len=1023, processor_channel="mono", **backend_kwargs):
        super().__init__()
        self.fir_len = fir_len
        self.conv = FIRConvolution(fir_len=fir_len, **backend_kwargs)
        
        if processor_channel == "midside":
            self.num_channels = 2
            self.process = self._process_midside
        elif processor_channel == "stereo":
            self.num_channels = 2
            self.process = self._process_mono_stereo
        elif processor_channel == "mono":
            self.num_channels = 1
            self.process = self._process_mono_stereo
        else:
            raise ValueError(f"Unknown channel type: {processor_channel}")
        # match self.processor_channel:
        #     case "midside":
        #         self.num_channels = 2
        #         self.process = self._process_midside
        #     case "stereo":
        #         self.num_channels = 2
        #         self.process = self._process_mono_stereo
        #     case "mono":
        #         self.num_channels = 1
        #         self.process = self._process_mono_stereo
        #     case _:
        #         raise ValueError(f"Unknown channel type: {self.channel}")

    def forward(self, input_signals, fir):
        r"""
        Performs the convolution operation.

        Args:
            input_signals (:python:`FloatTensor`, :math:`B \times C_\mathrm{in} \times L_\mathrm{in}`):
                A batch of input audio signals.
            fir (:python:`FloatTensor`, :math:`B \times C_\mathrm{filter} \times L_\mathrm{filter}`):
                A batch of FIR filters.

        Returns:
            :python:`FloatTensor`: A batch of convolved signals of shape :math:`B \times C_\mathrm{out} \times L_\mathrm{in}` where :math:`C_\mathrm{out} = \max (C_\mathrm{in}, C_\mathrm{filter})`.
        """
        fir = torch.tanh(fir)
        output_signals = self.process(input_signals, fir)
        return output_signals

    def _process_mono_stereo(self, input_signals, fir):
        fir = normalize_impulse(fir)
        return self.conv(input_signals, fir)

    def _process_midside(self, input_signals, fir):
        fir = normalize_impulse(fir)
        input_signals = lr_to_ms(input_signals)
        output_signals = self.conv(input_signals, fir)
        return ms_to_lr(output_signals)
