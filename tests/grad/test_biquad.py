import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.filters import BiquadFilter


# Create a BiquadFilter instance
proc = BiquadFilter(
    sample_rate=SAMPLE_RATE,
    filter_type='lowpass',
).to(DEVICE)


test_grad_batch(proc)

