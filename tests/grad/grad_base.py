import torch
import torch.nn as nn
import torch.nn.functional as F

SINGLE_BATCH = 1 
MONO_CHANNEL = 1 
BATCH_SIZE = 16
STEREO_CHANNEL = 2
NUM_SAMPLES = 44100*2
SAMPLE_RATE = 44100
DEVICE = 'cuda'


def check_valid(y, x, pann=False):
    print('> output requires_grad: ', y.requires_grad) 
    if pann:
        assert x.shape == (y.shape[0], 1, y.shape[2]), print(f'input: {x.shape}, output: {y.shape}')
    else:
        assert x.shape == y.shape, print(f'input: {x.shape}, output: {y.shape}')
    assert y.requires_grad == True

def test_grad_batch(proc, pann=False):
    print('> ====== START TEST GRAD BATCH ====== < ')
    dummy_condition = torch.randn(BATCH_SIZE, 1).to(DEVICE)
    proc = proc.to(DEVICE)
    controller = nn.Sequential(
        nn.Linear(1, 8),
        nn.LeakyReLU(0.2),
        nn.Linear(8, proc.count_num_parameters())
    ).to(DEVICE)
    test_signal = torch.randn(BATCH_SIZE, STEREO_CHANNEL, NUM_SAMPLES).to(DEVICE)
    y = proc(test_signal, controller(dummy_condition))
    loss = y.mean()
    loss.backward()

    check_valid(y, test_signal, pann=pann)
    print('> ====== TEST GRAD BATCH PASS ====== < ')


def test_grad_single(proc, pann=False):
    print('> ====== START TEST GRAD SINGLE ====== < ')
    dummy_condition = torch.randn(SINGLE_BATCH, 1).to(DEVICE)
    proc = proc.to(DEVICE)
    controller = nn.Sequential(
        nn.Linear(1, 8),
        nn.LeakyReLU(0.2),
        nn.Linear(8, proc.count_num_parameters())
    ).to(DEVICE)

    
    test_signal = torch.randn(SINGLE_BATCH, STEREO_CHANNEL, NUM_SAMPLES).to(DEVICE)
    y = proc(test_signal, controller(dummy_condition))
    loss = y.mean()
    loss.backward()

    check_valid(y, test_signal, pann=pann)
    print('> ====== TEST GRAD SINGLE PASS ====== < ')


def test_grad_mono(proc, pann=False):
    print('> ====== START TEST GRAD MONO ====== < ')
    dummy_condition = torch.randn(BATCH_SIZE, 1).to(DEVICE)
    proc = proc.to(DEVICE)
    controller = nn.Sequential(
        nn.Linear(1, 8),
        nn.LeakyReLU(0.2),
        nn.Linear(8, proc.count_num_parameters())
    ).to(DEVICE)

    test_signal = torch.randn(BATCH_SIZE, MONO_CHANNEL, NUM_SAMPLES).to(DEVICE)
    y = proc(test_signal, controller(dummy_condition))
    loss = y.mean()
    loss.backward()

    check_valid(y, test_signal, pann=pann)
    print('> ====== TEST GRAD MONO PASS ====== < ')

def test_dsp_stereo_params(proc, dsp_params, pann=False):
    print('> ====== START TEST STEREO DSP PARAMS ====== < ')
    proc = proc.to(DEVICE)
    
    test_signal = torch.randn(BATCH_SIZE, STEREO_CHANNEL, NUM_SAMPLES).to(DEVICE)
    y = proc(test_signal, None, dsp_params)
    assert y.shape == test_signal.shape
    print('> ====== TEST STEREO DSP PARAMS PASS ====== < ')

def test_dsp_mono_params(proc, dsp_params, pann=False):
    print('> ====== START TEST MONO DSP PARAMS ====== < ')
    proc = proc.to(DEVICE)
    
    test_signal = torch.randn(BATCH_SIZE, MONO_CHANNEL, NUM_SAMPLES).to(DEVICE)
    y = proc(test_signal, None, dsp_params)
    if pann:
        pass
    else:
        assert y.shape == test_signal.shape, print(f'input: {test_signal.shape}, output: {y.shape}')
    print('> ====== TEST MONO DSP PARAMS PASS ====== < ')


