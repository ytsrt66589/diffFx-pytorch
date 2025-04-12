import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.spatial import *
from diffFx_pytorch.processors import EffectParam

def test_panning():
    DSP_PARAMS = {
        'pan': 0.5
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = StereoPanning(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    # test_grad_batch(proc)
    # test_grad_single(proc)
    test_grad_mono(proc, pann=True)
    # test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS, pann=True)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = StereoPanning(
        sample_rate=SAMPLE_RATE,
        param_range={
            'pan': EffectParam(min_val=0.0, max_val=1.0)
        },
    ).to(DEVICE)

    # test_grad_batch(proc)
    # test_grad_single(proc)
    test_grad_mono(proc, pann=True)
    # test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    test_dsp_mono_params(proc, dsp_params=DSP_PARAMS, pann=True)

def test_widener():
    DSP_PARAMS = {
        'width': 0.5
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = StereoWidener(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    # test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    # test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = StereoWidener(
        sample_rate=SAMPLE_RATE,
        param_range={
            'width': EffectParam(min_val=0.0, max_val=1.0)
        },
    ).to(DEVICE)

    test_grad_batch(proc)   
    test_grad_single(proc)
    # test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    # test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

def test_imager():
    DSP_PARAMS = {
        'band0_width': 0.3,  # Reduce width in low frequencies
        'band1_width': 0.5,  # Keep mids unchanged
        'band2_width': 0.8,  # Enhance width in highs
        'crossover0_freq': 200.0,  # Low/mid crossover
        'crossover1_freq': 2000.0  # Mid/high crossover
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = StereoImager(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)  
    # test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    # test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = StereoImager(
        sample_rate=SAMPLE_RATE,
        param_range={
            'band0_width': EffectParam(min_val=0.0, max_val=1.0),
            'band1_width': EffectParam(min_val=0.0, max_val=1.0),
            'band2_width': EffectParam(min_val=0.0, max_val=1.0),
            'crossover0_freq': EffectParam(min_val=100.0, max_val=10000.0),
            'crossover1_freq': EffectParam(min_val=100.0, max_val=10000.0),
        },
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    # test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    # test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

def test_enhancer():
    DSP_PARAMS = {
        'delay_ms': 12.0,  # 12ms delay for natural width
        'width': 0.7      # 70% enhancement amount
    }
    print('> ====== TEST GRAD DEFAULT ====== < ')
    proc = StereoEnhancer(
        sample_rate=SAMPLE_RATE,
    ).to(DEVICE)

    test_grad_batch(proc)
    test_grad_single(proc)
    # test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    # test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)

    print('> ====== TEST GRAD PARAM RANGE ====== < ')
    proc = StereoEnhancer(
        sample_rate=SAMPLE_RATE,
        param_range={
            'delay_ms': EffectParam(min_val=10.0, max_val=3000.0),
            'width': EffectParam(min_val=0.0, max_val=1.0),
        },
    ).to(DEVICE)        
    test_grad_batch(proc)
    test_grad_single(proc)
    # test_grad_mono(proc)
    test_dsp_stereo_params(proc, dsp_params=DSP_PARAMS)
    # test_dsp_mono_params(proc, dsp_params=DSP_PARAMS)


if __name__ == '__main__':
    test_panning()
    test_widener()
    test_imager()
    test_enhancer()