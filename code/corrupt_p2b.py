"""P2b unseen corruptions (never used while designing the generic detector): blur and dim.  Same window/semantics as corrupt.py.
  blur [sigma_px] : cv2.GaussianBlur of every camera frame (ksize derived from sigma), sigma in pixels (frame 480x640)
  dim  [g]        : x' = round(g * x), brightness/contrast drop, no noise
Everything else is delegated to corrupt.Corruptor."""
import cv2
import numpy as np
from corrupt import Corruptor, parse_param as _parse, tag as _tag

KINDS2 = ("blur", "dim")


def parse_param(kind, param):
    return (float(param),) if kind in KINDS2 else _parse(kind, param)


def tag(kind, par):
    return f"{kind}{par[0]:g}" if kind in KINDS2 else _tag(kind, par)


class Corruptor2:
    def __init__(self, kind, par, t0, L, env_seed):
        self.kind, self.par, self.t0, self.L = kind, par, t0, L
        self.flicker_on = False
        self.base = None if kind in KINDS2 else Corruptor(kind, par, t0, L, env_seed)

    def active(self, t):
        return self.L > 0 and self.t0 <= t < self.t0 + self.L

    def __call__(self, t, obs):
        if self.base is not None:
            return self.base(t, obs)
        if not self.active(t):
            return obs
        out = {}
        for k, v in obs["pixels"].items():
            if self.kind == "blur":
                out[k] = cv2.GaussianBlur(v, (0, 0), sigmaX=self.par[0], borderType=cv2.BORDER_REFLECT)
            else:
                out[k] = np.rint(np.clip(self.par[0] * v.astype(np.float32), 0, 255)).astype(np.uint8)
        return {**obs, "pixels": out}
