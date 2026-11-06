"""P2 camera corruptions applied only to the image given to the policy, only inside the window [t0, t0+L).
agent_pos is never touched; outside the window the policy immediately sees the true current image again.

kinds (param in brackets)
  black                 all-zero image                     (identical to blackout.Blackout, T2)
  freeze                last image before the window       (identical to blackout.Blackout, T2)
  noise   [sigma]       x' = clip(x/255 + N(0, sigma^2), 0, 1); fresh noise every step.        sigma in [0,1] image scale
  occlude [area,pos]    black rectangle covering `area` of the frame (sides scaled by sqrt(area), same aspect ratio);
                        pos = 'center' (frame centre) or 'rand' (centre uniform in the central 50% x 50% of the frame,
                        drawn once per episode from env_seed).
  delay   [d]           step t shows the true image of step max(t-d, 0) (stale but otherwise normal frames).
  flicker [p]           every step in the window, with probability p, the frame is replaced by a dim noisy frame
                        clip(0.2*x/255 + N(0, 0.05^2), 0, 1) (low brightness + noise, NOT black); otherwise the true frame.
All randomness comes from numpy Generators seeded by (20000+env_seed, t), so the same env_seed/t gives the same noise
realisation under every rule (paired design); the pixel values underneath still depend on the trajectory.
"""
from collections import deque
import numpy as np
from blackout import Blackout

KINDS = ("black", "freeze", "noise", "occlude", "delay", "flicker")
FLICKER_GAIN, FLICKER_SIGMA = 0.2, 0.05
SALT = 20000


def parse_param(kind, param):
    """CLI param string -> tuple. occlude: '0.25:center'; others: float (or int for delay)."""
    if kind in ("black", "freeze"):
        return ()
    if kind == "occlude":
        a, pos = (param.split(":") + ["center"])[:2]
        assert pos in ("center", "rand")
        return (float(a), pos)
    if kind == "delay":
        return (int(param),)
    return (float(param),)


def tag(kind, par):
    if kind in ("black", "freeze"):
        return kind
    if kind == "occlude":
        return f"occlude{par[0]:g}{par[1]}"
    return f"{kind}{par[0]:g}"


class Corruptor:
    def __init__(self, kind, par, t0, L, env_seed):
        assert kind in KINDS
        self.kind, self.par, self.t0, self.L, self.seed = kind, par, t0, L, env_seed
        self.bo = Blackout(t0, L, kind) if kind in ("black", "freeze") else None
        self.hist = deque(maxlen=(par[0] + 1) if kind == "delay" else 1)
        self.flicker_on = False  # was this step's frame replaced by a flicker frame (for logging)
        if kind == "occlude":
            area, pos = par
            self.oh_frac = self.ow_frac = float(np.sqrt(area))
            if pos == "center":
                self.c = (0.5, 0.5)
            else:
                r = np.random.default_rng([SALT, env_seed, 1])
                self.c = (float(r.uniform(0.25, 0.75)), float(r.uniform(0.25, 0.75)))

    def active(self, t):
        return self.L > 0 and self.t0 <= t < self.t0 + self.L

    def __call__(self, t, obs):
        if self.bo is not None:
            return self.bo(t, obs)
        pix = obs["pixels"]
        if self.kind == "delay":
            self.hist.append({k: v.copy() for k, v in pix.items()})  # hist[-1] = step t
        self.flicker_on = False
        if not self.active(t):
            return obs
        rng = np.random.default_rng([SALT, self.seed, 2 + t])
        out = {}
        for k, v in pix.items():
            if self.kind == "noise":
                x = v.astype(np.float32) / 255.0 + rng.normal(0.0, self.par[0], v.shape).astype(np.float32)
                out[k] = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
            elif self.kind == "occlude":
                H, W = v.shape[:2]
                h, w = int(round(H * self.oh_frac)), int(round(W * self.ow_frac))
                cy, cx = int(round(self.c[0] * H)), int(round(self.c[1] * W))
                y0 = min(max(cy - h // 2, 0), H - h); x0 = min(max(cx - w // 2, 0), W - w)
                o = v.copy(); o[y0:y0 + h, x0:x0 + w] = 0; out[k] = o
            elif self.kind == "delay":
                out[k] = self.hist[0][k]  # oldest kept frame = true image of step max(t-d, 0)
            elif self.kind == "flicker":
                if rng.random() < self.par[0]:
                    x = FLICKER_GAIN * v.astype(np.float32) / 255.0 + rng.normal(0.0, FLICKER_SIGMA, v.shape).astype(np.float32)
                    out[k] = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
                    self.flicker_on = True
                else:
                    out[k] = v
        return {**obs, "pixels": out}
