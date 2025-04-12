import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.gain.gain import Gain
from diffFx_pytorch.processors import EffectParam

print('> ====== TEST GRAD DEFAULT ====== < ')
# Create a Gain instance
proc = Gain(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, {'gain_db': 6.0})
test_dsp_mono_params(proc, {'gain_db': 6.0})

print('> ====== TEST GRAD PARAM RANGE ====== < ')
# Create a Gain instance
proc = Gain(
    sample_rate=SAMPLE_RATE,
    param_range={
        'gain_db': EffectParam(min_val=-24.0, max_val=24.0)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, {'gain_db': 6.0})
test_dsp_mono_params(proc, {'gain_db': 6.0})
