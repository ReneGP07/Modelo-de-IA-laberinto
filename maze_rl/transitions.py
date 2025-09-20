
import numpy as np
import torch
from .constants import ACCIONES
from .maze import estado_idx

def precompute_transitions(lab, inicio, meta, r_step=-1.0, r_wall=-5.0, r_goal=100.0):
    """Precálculo de transiciones deterministas y recompensas."""
    h,w = lab.shape
    n_states = h*w; n_actions = len(ACCIONES)
    next_s = np.zeros((n_states, n_actions), dtype=np.int64)
    reward = np.zeros((n_states, n_actions), dtype=np.float32)
    done = np.zeros((n_states, n_actions), dtype=np.bool_)
    goal_idx = estado_idx(lab.shape, meta)

    for y in range(h):
        for x in range(w):
            s_idx = y*w + x
            for a,(dy,dx) in enumerate(ACCIONES):
                ny, nx = y+dy, x+dx
                if not (0 <= ny < h and 0 <= nx < w) or lab[ny,nx] == 1:
                    next_s[s_idx,a] = s_idx; reward[s_idx,a] = r_wall; done[s_idx,a] = False
                else:
                    ns = ny*w + nx; next_s[s_idx,a] = ns
                    if ns == goal_idx:
                        reward[s_idx,a] = r_goal; done[s_idx,a] = True
                    else:
                        reward[s_idx,a] = r_step; done[s_idx,a] = False

    start_idx = estado_idx(lab.shape, inicio)
    return (torch.from_numpy(next_s),
            torch.from_numpy(reward),
            torch.from_numpy(done),
            torch.tensor(start_idx, dtype=torch.long),
            torch.tensor(goal_idx, dtype=torch.long))
