import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.modulation import Chorus, MultiVoiceChorus, StereoChorus, Flanger, FeedbackFlanger, StereoFlanger    
from diffFx_pytorch.processors import EffectParam

DSP_PARAMS = {
    'delay_ms': 20.0,  # 20ms base delay
    'rate': 2.0,       # 2 Hz modulation
    'depth': 0.15,     # Moderate intensity
    'mix': 0.5         # Equal mix
}
print('> ====== TEST GRAD DEFAULT ====== < ')
# Create a Gain instance
proc = Chorus(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
test_dsp_mono_params(proc, DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
# Create a Gain instance
proc = Chorus(
    sample_rate=SAMPLE_RATE,
    param_range={
        'depth': EffectParam(min_val=0.0, max_val=100.0),
        'rate': EffectParam(min_val=0.0, max_val=10.0),
        'mix': EffectParam(min_val=0.0, max_val=1.0)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
test_dsp_mono_params(proc, DSP_PARAMS)


DSP_PARAMS = {
    'delay_ms': 5.0,    # Base delay
    'rate': 1.5,        # Modulation rate
    'depth': 0.15,      # Moderate detuning
    'mix': 0.7,         # Mostly wet
    'g0': 1.0,          # Full level voice 1
    'g1': 0.8,          # Reduced voice 2
}
print('> ====== TEST GRAD DEFAULT ====== < ')
# Create a Gain instance
proc = MultiVoiceChorus(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
test_dsp_mono_params(proc, DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
# Create a Gain instance
proc = MultiVoiceChorus(
    sample_rate=SAMPLE_RATE,
    param_range={
        'delay_ms': EffectParam(min_val=0.0, max_val=100.0),
        'rate': EffectParam(min_val=0.0, max_val=10.0),
        'depth': EffectParam(min_val=0.0, max_val=1.0),
        'mix': EffectParam(min_val=0.0, max_val=1.0),
        'g0': EffectParam(min_val=0.0, max_val=1.0),
        'g1': EffectParam(min_val=0.0, max_val=1.0)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
test_dsp_mono_params(proc, DSP_PARAMS)

DSP_PARAMS = {
    'delay_ms': 5.0,    # Base delay
    'rate': 1.5,        # Modulation rate
    'depth': 0.15,      # Moderate detuning
    'mix': 0.7,         # Mostly wet
    'g0': 1.0,          # Full voice 1
    'pan0': -0.7,       # Voice 1 left
    'g1': 0.8,          # Reduced voice 2
    'pan1': 0.7         # Voice 2 right
}
print('> ====== TEST GRAD DEFAULT ====== < ')
# Create a Gain instance
proc = StereoChorus(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)  
# test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
# test_dsp_mono_params(proc, DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
# Create a Gain instance
proc = StereoChorus(
    sample_rate=SAMPLE_RATE,
    param_range={
        'delay_ms': EffectParam(min_val=0.0, max_val=100.0),
        'rate': EffectParam(min_val=0.0, max_val=10.0),
        'depth': EffectParam(min_val=0.0, max_val=1.0),
        'mix': EffectParam(min_val=0.0, max_val=1.0),
        'g0': EffectParam(min_val=0.0, max_val=1.0),
        'pan0': EffectParam(min_val=-1.0, max_val=1.0),
        'g1': EffectParam(min_val=0.0, max_val=1.0),
        'pan1': EffectParam(min_val=-1.0, max_val=1.0)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
# test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
# test_dsp_mono_params(proc, DSP_PARAMS)






DSP_PARAMS = {
    'delay_ms': 1.0,    # Increased range
    'rate': 0.1,         # More musical range
    'depth': 0.5,        # Full range
    'mix': 0.5
}
print('> ====== TEST GRAD DEFAULT ====== < ')
# Create a Gain instance
proc = Flanger(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
test_dsp_mono_params(proc, DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
# Create a Gain instance
proc = Flanger(
    sample_rate=SAMPLE_RATE,
    param_range={
        'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
        'rate': EffectParam(min_val=0.1, max_val=2.0),         # More musical range
        'depth': EffectParam(min_val=0.0, max_val=1.0),        # Full range
        'mix': EffectParam(min_val=0.0, max_val=1.0)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
test_dsp_mono_params(proc, DSP_PARAMS)


DSP_PARAMS = {
    'delay_ms': 1.0,    # Increased range
    'rate': 0.1,         # More musical range
    'depth': 0.5,        # Full range
    'feedback': 0.5,
    'mix': 0.5
}
print('> ====== TEST GRAD DEFAULT ====== < ')
# Create a Gain instance
proc = FeedbackFlanger(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
test_dsp_mono_params(proc, DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
# Create a Gain instance
proc = FeedbackFlanger(
    sample_rate=SAMPLE_RATE,
    param_range={
        'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
        'rate': EffectParam(min_val=0.1, max_val=10.0),         # More musical range
        'depth': EffectParam(min_val=0.0, max_val=0.25),        # Full range
        'feedback': EffectParam(min_val=0.0, max_val=0.7),
        'mix': EffectParam(min_val=0.0, max_val=1.0)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
test_dsp_mono_params(proc, DSP_PARAMS)

DSP_PARAMS = {
    'delay_ms': 1.0,    # Increased range
    'rate': 0.1,         # More musical range
    'depth': 0.5,        # Full range
    'mix': 0.5
}
print('> ====== TEST GRAD DEFAULT ====== < ')
# Create a Gain instance
proc = StereoFlanger(
    sample_rate=SAMPLE_RATE,
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)  
# test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
# test_dsp_mono_params(proc, DSP_PARAMS)

print('> ====== TEST GRAD PARAM RANGE ====== < ')
# Create a Gain instance
proc = StereoFlanger(
    sample_rate=SAMPLE_RATE,
    param_range={
        'delay_ms': EffectParam(min_val=1.0, max_val=10.0),    # Increased range
        'rate': EffectParam(min_val=0.1, max_val=2.0),         # More musical range
        'depth': EffectParam(min_val=0.0, max_val=1.0),        # Full range
        'mix': EffectParam(min_val=0.0, max_val=1.0)
    }
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
# test_grad_mono(proc)
test_dsp_stereo_params(proc, DSP_PARAMS)
# test_dsp_mono_params(proc, DSP_PARAMS)




