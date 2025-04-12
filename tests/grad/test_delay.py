import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.delay import *
from diffFx_pytorch.processors import EffectParam

def test_basic_delay():
    DSP_PARAMS = {
        'delay_ms': 100.0,
        'mix': 0.5
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = BasicDelay(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = BasicDelay(
        sample_rate=SAMPLE_RATE,
        param_range={
            'delay_ms': EffectParam(min_val=10.0, max_val=3000.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0)
        },
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

def test_feedback_delay():
    DSP_PARAMS = {
        'delay_ms': 100.0,
        'mix': 0.5,
        'fb_gain': 0.5,
        'ff_gain': 0.5
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = BasicFeedbackDelay(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = BasicFeedbackDelay(
        sample_rate=SAMPLE_RATE,
        param_range={
            'delay_ms': EffectParam(min_val=10.0, max_val=3000.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0),
            'fb_gain': EffectParam(min_val=0.0, max_val=0.99),
            'ff_gain': EffectParam(min_val=0.0, max_val=0.99)
        },
    ).to(DEVICE)

    test_grad_batch(proc)   
    test_grad_single(proc)
    test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

def test_slapback_delay():
    DSP_PARAMS = {
        'delay_ms': 100.0,
        'mix': 0.5,
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = SlapbackDelay(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)  
    test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = SlapbackDelay(
        sample_rate=SAMPLE_RATE,
        param_range={
            'delay_ms': EffectParam(min_val=10.0, max_val=3000.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0),
        },
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

def test_pingpong_delay():
    DSP_PARAMS = {
        'delay_ms': 100.0,
        'mix': 0.5,
        'feedback_ch1': 0.5,
        'feedback_ch2': 0.5,
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = PingPongDelay(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    # test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    # test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = PingPongDelay(
        sample_rate=SAMPLE_RATE,
        param_range={
            'delay_ms': EffectParam(min_val=10.0, max_val=3000.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0),
            'feedback_ch1': EffectParam(min_val=0.0, max_val=0.99),
            'feedback_ch2': EffectParam(min_val=0.0, max_val=0.99),
        },
    ).to(DEVICE)        
    test_grad_batch(proc)
    test_grad_single(proc)
    # test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    # test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

def test_multitap_delay():
    DSP_PARAMS = {
        '0_tap_delays_ms': 125.0,  # Eighth note at 120 BPM
        '0_tap_gains': 0.8,
        '1_tap_delays_ms': 250.0,  # Quarter note
        '1_tap_gains': 0.6,
        '2_tap_delays_ms': 375.0,  # Dotted quarter
        '2_tap_gains': 0.4,
        '3_tap_delays_ms': 500.0,  # Half note
        '3_tap_gains': 0.2,
        'mix': 0.5
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = MultiTapsDelay(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = MultiTapsDelay(
        sample_rate=SAMPLE_RATE,
        param_range={
            '0_tap_delays_ms': EffectParam(min_val=10.0, max_val=3000.0),
            '0_tap_gains': EffectParam(min_val=0.0, max_val=1.0),
            '1_tap_delays_ms': EffectParam(min_val=10.0, max_val=3000.0),
            '1_tap_gains': EffectParam(min_val=0.0, max_val=1.0),
            '2_tap_delays_ms': EffectParam(min_val=10.0, max_val=3000.0),
            '2_tap_gains': EffectParam(min_val=0.0, max_val=1.0),
            '3_tap_delays_ms': EffectParam(min_val=10.0, max_val=3000.0),
            '3_tap_gains': EffectParam(min_val=0.0, max_val=1.0),
            'mix': EffectParam(min_val=0.0, max_val=1.0),
        },
    ).to(DEVICE)    
    test_grad_batch(proc)
    test_grad_single(proc)
    test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

if __name__ == '__main__':
    test_basic_delay()
    test_feedback_delay()
    test_slapback_delay()
    test_pingpong_delay()
    test_multitap_delay()