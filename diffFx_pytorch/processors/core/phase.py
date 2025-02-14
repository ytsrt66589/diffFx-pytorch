import torch
import numpy as np 

def unwrap_phase(phase, dim=-1):
    dd = torch.diff(phase, dim=dim)
    ddmod = torch.remainder(dd + np.pi, 2 * np.pi) - np.pi
    ddmod = torch.where((ddmod == -np.pi) & (dd > 0), torch.tensor(np.pi, device=phase.device, dtype=phase.dtype), ddmod)
    phase_unwrapped = torch.cumsum(torch.cat([phase[..., 0:1], ddmod], dim=dim), dim=dim)
    return phase_unwrapped