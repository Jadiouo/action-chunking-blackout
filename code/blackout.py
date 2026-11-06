"""Camera blackout applied only to the observation fed to the policy."""
import numpy as np


class Blackout:
    """mode: 'black' (zero image) or 'freeze' (last frame before blackout).
    Active for steps t in [t0, t0+L). agent_pos is never touched."""

    def __init__(self, t0, L, mode="black"):
        assert mode in ("black", "freeze")
        self.t0, self.L, self.mode = t0, L, mode
        self._last = None

    def active(self, t):
        return self.L > 0 and self.t0 <= t < self.t0 + self.L

    def __call__(self, t, obs):
        """obs: gym-aloha dict {'pixels': {'top': HxWx3 uint8}, 'agent_pos': ...}. Returns modified copy."""
        if not self.active(t):
            self._last = {k: v.copy() for k, v in obs["pixels"].items()}
            return obs
        if self.mode == "black":
            pix = {k: np.zeros_like(v) for k, v in obs["pixels"].items()}
        else:
            pix = self._last if self._last is not None else {k: v.copy() for k, v in obs["pixels"].items()}
        return {**obs, "pixels": pix}
