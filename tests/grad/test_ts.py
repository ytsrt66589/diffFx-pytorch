import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.eq.tonestack import Tonestack
from diffFx_pytorch.processors import EffectParam

print('> ====== TEST GRAD DEFAULT ====== < ')
proc = Tonestack(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, {'bass': 0.5, 'mid': 0.5, 'treble': 0.5})
test_dsp_mono_params(proc, {'bass': 0.5, 'mid': 0.5, 'treble': 0.5})

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = Tonestack(
    sample_rate=SAMPLE_RATE,
    param_range={
        'bass': EffectParam(min_val=0.0, max_val=1.0),
        'mid': EffectParam(min_val=0.0, max_val=1.0),
        'treble': EffectParam(min_val=0.0, max_val=1.0)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, {'bass': 0.5, 'mid': 0.5, 'treble': 0.5})
test_dsp_mono_params(proc, {'bass': 0.5, 'mid': 0.5, 'treble': 0.5})
