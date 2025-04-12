import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.dynamics.limiter import MultiBandLimiter
from diffFx_pytorch.processors import EffectParam

DSP_PARAMS = {
    'band0_threshold_db': -24.0,  # Low band
    'band0_ratio': 4.0,
    'band0_knee_db': 6.0,
    'band0_attack_ms': 10.0,
    'band0_release_ms': 100.0,
    'band0_makeup_db': 3.0,
    'band1_threshold_db': -18.0,  # Mid band
    'band1_ratio': 3.0,
    'band1_knee_db': 6.0,
    'band1_attack_ms': 5.0,
    'band1_release_ms': 50.0,
    'band1_makeup_db': 2.0,
    'band2_threshold_db': -12.0,  # High band
    'band2_ratio': 2.0,
    'band2_knee_db': 6.0,
    'band2_attack_ms': 1.0,
    'band2_release_ms': 20.0,
    'band2_makeup_db': 1.0,
    'crossover0_freq': 200.0,     # Low-Mid split
    'crossover1_freq': 2000.0     # Mid-High split
}   
print('> ====== TEST GRAD DEFAULT ====== < ')
proc = MultiBandLimiter(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = MultiBandLimiter(
    sample_rate=SAMPLE_RATE,
    param_range={
        'band0_threshold_db': EffectParam(min_val=-60.0, max_val=0.0),
        'band0_ratio': EffectParam(min_val=1.0, max_val=20.0),
        'band0_knee_db': EffectParam(min_val=0.0, max_val=12.0),
        'band0_attack_ms': EffectParam(min_val=0.1, max_val=100.0),
        'band0_release_ms': EffectParam(min_val=10.0, max_val=1000.0),
        'band0_makeup_db': EffectParam(min_val=-12.0, max_val=12.0),
        'band1_threshold_db': EffectParam(min_val=-60.0, max_val=0.0),
        'band1_ratio': EffectParam(min_val=1.0, max_val=20.0),
        'band1_knee_db': EffectParam(min_val=0.0, max_val=12.0),
        'band1_attack_ms': EffectParam(min_val=0.1, max_val=100.0),
        'band1_release_ms': EffectParam(min_val=10.0, max_val=1000.0),
        'band1_makeup_db': EffectParam(min_val=-12.0, max_val=12.0) 
    },
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)
