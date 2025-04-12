import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.eq.graphicEQ import GraphicEqualizer
from diffFx_pytorch.processors import EffectParam

print('> ====== TEST GRAD DEFAULT ====== < ')
proc = GraphicEqualizer(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, {
    f'band_{i+1}_gain_db': 2.0 for i in range(proc.num_bands)
})
test_dsp_mono_params(proc, {
    f'band_{i+1}_gain_db': 2.0 for i in range(proc.num_bands)
})

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = GraphicEqualizer(
    sample_rate=SAMPLE_RATE,
    param_range={
        f'band_{i+1}_gain_db': EffectParam(min_val=-12.0, max_val=12.0) for i in range(proc.num_bands)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, {
    f'band_{i+1}_gain_db': 2.0 for i in range(proc.num_bands)
})
test_dsp_mono_params(proc, {
    f'band_{i+1}_gain_db': 2.0 for i in range(proc.num_bands)
})
