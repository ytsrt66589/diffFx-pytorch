import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.distortion.bitcrusher import BitCrusher
from diffFx_pytorch.processors import EffectParam

DSP_PARAMS = {
    'bit_depth': 2.0,
}
print('> ====== TEST GRAD DEFAULT ====== < ')
proc = BitCrusher(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = BitCrusher(
    sample_rate=SAMPLE_RATE,
    param_range={
        'bit_depth': EffectParam(min_val=1.0, max_val=16.0),
    },
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)
