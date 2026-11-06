"""lerobot-train entry point with two additions over lerobot 0.4.4 (everything else unchanged):

1. pin_memory disabled in the DataLoader.  The first dp_v1 run was OOM-killed (12G cgroup) after ~2600 steps; one float32
   batch is 64x2x3x480x640x4 B = 470 MB and pinned host buffers add to the footprint.
2. Optional EMA of the diffusion policy weights (lerobot 0.4.4's DiffusionPolicy has none; the original DP uses EMA).
   Enabled when env DP_EMA_DECAY is set (e.g. 0.999).  decay_t = min(DP_EMA_DECAY, (1+t)/(10+t)) (standard warm-up).
   lerobot calls policy.update() after every optimizer step (the hook it uses for TD-MPC's target network), which we use
   to update the EMA.  The EMA weights are stored next to every checkpoint as `pretrained_model/ema.safetensors`
   (including the EMA step) and are restored on --resume.  model.safetensors keeps the raw (non-EMA) weights.
"""
import os
import sys
from pathlib import Path

import torch
import torch.utils.data as tud

_DL = tud.DataLoader


class DataLoaderNoPin(_DL):
    def __init__(self, *a, **kw):
        kw["pin_memory"] = False
        super().__init__(*a, **kw)


tud.DataLoader = DataLoaderNoPin

from safetensors.torch import load_file, save_file  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402

EMA_FILE = "ema.safetensors"
STEP_KEY = "__ema_step__"
_DECAY = float(os.environ["DP_EMA_DECAY"]) if os.environ.get("DP_EMA_DECAY") else None


def _ema_init(self):
    sd = {k: v.detach().clone() for k, v in self.state_dict().items()}
    self._ema, self._ema_step = sd, 0
    pp = getattr(self.config, "pretrained_path", None)
    if pp and (Path(pp) / EMA_FILE).exists():  # resume
        loaded = load_file(str(Path(pp) / EMA_FILE), device=str(next(self.parameters()).device))
        self._ema_step = int(loaded.pop(STEP_KEY).item())
        for k, v in loaded.items():
            sd[k].copy_(v)
        print(f"[EMA] restored from {pp}/{EMA_FILE} at ema_step={self._ema_step}", flush=True)
    else:
        print(f"[EMA] initialised from current weights, decay={_DECAY}", flush=True)


@torch.no_grad()
def _update(self):
    if _DECAY is None:
        return
    if getattr(self, "_ema", None) is None:
        _ema_init(self)
    self._ema_step += 1
    d = min(_DECAY, (1 + self._ema_step) / (10 + self._ema_step))
    for k, v in self.state_dict().items():
        e = self._ema[k]
        if v.dtype.is_floating_point:
            e.mul_(d).add_(v.detach(), alpha=1 - d)
        else:
            e.copy_(v)


_orig_save = DiffusionPolicy._save_pretrained


def _save_pretrained(self, save_directory):
    _orig_save(self, save_directory)
    if _DECAY is not None and getattr(self, "_ema", None) is not None:
        d = {k: v.contiguous().cpu() for k, v in self._ema.items()}
        d[STEP_KEY] = torch.tensor(self._ema_step)
        save_file(d, str(Path(save_directory) / EMA_FILE))


DiffusionPolicy.update = _update
DiffusionPolicy._save_pretrained = _save_pretrained

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
