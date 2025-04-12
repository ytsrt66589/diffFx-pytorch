import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.distortion.func import *
from diffFx_pytorch.processors import EffectParam

DSP_PARAMS = {
    'mix': 0.5,
    'pre_gain_db': 0.0,
    'post_gain_db': 0.0,
    'dc_bias': 0.0,
}
print('> ====== TEST GRAD DEFAULT ====== < ')
proc = TanHDist(
    sample_rate=SAMPLE_RATE,
    shaping_mode=True,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = TanHDist(
    sample_rate=SAMPLE_RATE,
    param_range={
        'mix': EffectParam(min_val=0.0, max_val=1.0),
        'pre_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        'post_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        'dc_bias': EffectParam(min_val=-0.2, max_val=0.2),
    },
    shaping_mode=True,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)




DSP_PARAMS = {
    'mix': 0.5,
    'pre_gain_db': 0.0,
    'post_gain_db': 0.0,
    'dc_bias': 0.0,
}
print('> ====== TEST GRAD DEFAULT ====== < ')
proc = SoftDist(
    sample_rate=SAMPLE_RATE,
    shaping_mode=True,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = SoftDist(
    sample_rate=SAMPLE_RATE,
    param_range={
        'mix': EffectParam(min_val=0.0, max_val=1.0),
        'pre_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        'post_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        'dc_bias': EffectParam(min_val=-0.2, max_val=0.2),
    },
    shaping_mode=True,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)



DSP_PARAMS = {
    'mix': 0.5,
    'pre_gain_db': 0.0,
    'post_gain_db': 0.0,
    'dc_bias': 0.0,
    'threshold': 0.5,
}
print('> ====== TEST GRAD DEFAULT ====== < ')
proc = HardDist(
    sample_rate=SAMPLE_RATE,
    shaping_mode=True,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = HardDist(
    sample_rate=SAMPLE_RATE,
    param_range={
        'mix': EffectParam(min_val=0.0, max_val=1.0),
        'pre_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        'post_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        'dc_bias': EffectParam(min_val=-0.2, max_val=0.2),
        'threshold': EffectParam(min_val=0.1, max_val=1.0),
    },
    shaping_mode=True,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)
