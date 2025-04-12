import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.dynamics.noisegate import NoiseGate
from diffFx_pytorch.processors import EffectParam

DSP_PARAMS = {
    'threshold_db': -20.0,
    'range_db': 0.0,
    'knee_db': 0.0,
    'attack_ms': 0.1,
    'release_ms': 10.0,
}
print('> ====== TEST GRAD DEFAULT ====== < ')
proc = NoiseGate(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = NoiseGate(
    sample_rate=SAMPLE_RATE,
    param_range={
        'threshold_db': EffectParam(min_val=-90.0, max_val=-20.0),
        'range_db': EffectParam(min_val=-90.0, max_val=0.0),
        'knee_db': EffectParam(min_val=0.0, max_val=6.0),
        'attack_ms': EffectParam(min_val=0.1, max_val=20.0),
        'release_ms': EffectParam(min_val=10.0, max_val=1000.0),
    },
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)
