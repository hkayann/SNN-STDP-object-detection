# stimuli.py
import torch
from utils import device, cy, cx

@torch.no_grad()
def make_center_impulse(h, w):
    x = torch.zeros(1, 1, h, w, device=device)
    x[0, 0, cy, cx] = 1.0
    return x

@torch.no_grad()
def make_impulse_at_x(h, w, xcol):
    xcol = max(0, min(w-1, xcol))
    img = torch.zeros(1, 1, h, w, device=device)
    img[0, 0, cy, xcol] = 1.0
    return img

@torch.no_grad()
def make_center_minibar(h, w):
    x = torch.zeros(1, 1, h, w, device=device)
    x[0, 0, cy-1, cx] = 1.0
    x[0, 0, cy,   cx] = 1.0
    x[0, 0, cy+1, cx] = 1.0
    return x

@torch.no_grad()
def make_minibar_at_x(h, w, xcol):
    xcol = max(1, min(w-2, xcol))  # keep 3-pixel vertical bar inside
    x = torch.zeros(1, 1, h, w, device=device)
    x[0, 0, cy-1, xcol] = 1.0
    x[0, 0, cy,   xcol] = 1.0
    x[0, 0, cy+1, xcol] = 1.0
    return x

@torch.no_grad()
def make_random_impulse(h, w, rng=None):
    import random
    r = rng if rng is not None else random
    y = r.randrange(0, h)
    x = r.randrange(0, w)
    img = torch.zeros(1, 1, h, w, device=device)
    img[0, 0, y, x] = 1.0
    return img