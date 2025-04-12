import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.eq.parametricEQ import ParametricEqualizer
from diffFx_pytorch.processors import EffectParam

print('> ====== TEST GRAD DEFAULT ====== < ')
proc = ParametricEqualizer(
    sample_rate=SAMPLE_RATE,
    num_peak_filters=1, # 1 peak filter
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params={
    # Low shelf parameters
    'low_shelf_gain_db': 0.0,
    'low_shelf_frequency': 100.0,
    'low_shelf_q_factor': 0.707,
    f'peak_1_gain_db': 0.0,
    f'peak_1_frequency': 1000.0,
    f'peak_1_q_factor': 0.707,
    # High shelf parameters
    'high_shelf_gain_db': 0.0,
    'high_shelf_frequency': 10000.0,
    'high_shelf_q_factor': 0.707,
})
test_dsp_mono_params(proc, dsp_params={
    # Low shelf parameters
    'low_shelf_gain_db': 0.0,
    'low_shelf_frequency': 100.0,
    'low_shelf_q_factor': 0.707,
    f'peak_1_gain_db': 0.0,
    f'peak_1_frequency': 1000.0,
    f'peak_1_q_factor': 0.707,
    # High shelf parameters
    'high_shelf_gain_db': 0.0,
    'high_shelf_frequency': 10000.0,
    'high_shelf_q_factor': 0.707,
})

print('> ====== TEST GRAD PARAM RANGE ====== < ')
proc = ParametricEqualizer(
    sample_rate=SAMPLE_RATE,
    param_range={
        # Low shelf parameters
        'low_shelf_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        'low_shelf_frequency': EffectParam(min_val=20.0, max_val=500.0),
        'low_shelf_q_factor': EffectParam(min_val=0.1, max_val=1.0),
        f'peak_1_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        f'peak_1_frequency': EffectParam(min_val=20.0, max_val=20000.0),
        f'peak_1_q_factor': EffectParam(min_val=0.1, max_val=10.0),
        # High shelf parameters
        'high_shelf_gain_db': EffectParam(min_val=-12.0, max_val=12.0),
        'high_shelf_frequency': EffectParam(min_val=5000.0, max_val=20000.0),
        'high_shelf_q_factor': EffectParam(min_val=0.1, max_val=1.0),
    },
    num_peak_filters=1, # 1 peak filter
).to(DEVICE)

test_grad_batch(proc)
test_grad_single(proc)
test_grad_mono(proc)
test_dsp_stereo_params(proc, dsp_params={
    # Low shelf parameters
    'low_shelf_gain_db': 0.0,
    'low_shelf_frequency': 100.0,
    'low_shelf_q_factor': 0.707,
    f'peak_1_gain_db': 0.0,
    f'peak_1_frequency': 1000.0,
    f'peak_1_q_factor': 0.707,
    # High shelf parameters
    'high_shelf_gain_db': 0.0,
    'high_shelf_frequency': 10000.0,
    'high_shelf_q_factor': 0.707,
})
test_dsp_mono_params(proc, dsp_params={
    # Low shelf parameters
    'low_shelf_gain_db': 0.0,
    'low_shelf_frequency': 100.0,
    'low_shelf_q_factor': 0.707,
    f'peak_1_gain_db': 0.0,
    f'peak_1_frequency': 1000.0,
    f'peak_1_q_factor': 0.707,
    # High shelf parameters
    'high_shelf_gain_db': 0.0,
    'high_shelf_frequency': 10000.0,
    'high_shelf_q_factor': 0.707,
})
