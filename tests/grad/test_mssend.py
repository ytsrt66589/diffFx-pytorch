import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_base import *
from diffFx_pytorch.processors.eq.parametricEQ import ParametricEqualizer
from diffFx_pytorch.processors.utilites import MidSideProc, SendProc
from diffFx_pytorch.processors import EffectParam

print('> ====== TEST GRAD DEFAULT ====== < ')
proc = ParametricEqualizer(
    sample_rate=SAMPLE_RATE,
    num_peak_filters=1, # 1 peak filter
).to(DEVICE)

ms_proc = MidSideProc(proc)
send_proc = SendProc(proc)

test_ms_send_grad_batch(ms_proc)
# test_grad_single(ms_proc)
# test_grad_mono(ms_proc)
test_ms_send_stereo_params(ms_proc, dsp_params={
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
