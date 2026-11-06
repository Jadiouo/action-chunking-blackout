"""CARE tau calibration on CLEAN seeds 100-119 (python3, numpy). Input: .scratch/p2/curve/n30_seed*.npy, each (decisions, 50)
= cumulative discrepancy R_i of the chunk queried every 30 steps (open loop, fixed n=30) against the previous chunk's unexecuted actions.
For a candidate tau, H_t = number of leading i with R_i <= tau (min 1); we pick tau so that the mean H on clean data hits
a target fraction of H_max=50: 0.4 / 0.6 / 0.8 (-> 20 / 30 / 40; 0.6 is the main value, following Table 3 of the CARE paper where clean
mean length was 44-76 % of H_max).  Open-loop curves approximate the closed loop (decision points differ), so run_p2p3.py logs the
realised mean H under each tau (care_meanH) and p2p3_notes.md reports it."""
import glob, json
import numpy as np

C = np.concatenate([np.load(f) for f in sorted(glob.glob(".scratch/p2/curve/n30_seed*.npy"))])
print("decisions", C.shape)
def meanH(tau):
    ok = C <= tau
    first_bad = np.where((~ok).any(1), (~ok).argmax(1), C.shape[1])
    return np.maximum(first_bad, 1).mean()
lo, hi = 1e-6, float(C.max()) * 2
out = {"decisions": int(C.shape[0]), "R_percentiles_at_i0": {q: float(np.percentile(C[:, 0], q)) for q in (50, 90, 99)},
       "R_percentiles_at_i49": {q: float(np.percentile(C[:, -1], q)) for q in (10, 50, 90)}, "tau": {}}
for frac in (0.4, 0.6, 0.8):
    tgt = frac * 50; a, b = lo, hi
    for _ in range(80):
        m = np.sqrt(a * b)
        if meanH(m) < tgt: a = m
        else: b = m
    out["tau"][str(frac)] = dict(target_meanH=tgt, tau=float(b), open_loop_meanH=float(meanH(b)))
json.dump(out, open("results/parts/care_tau_calib.json", "w"), indent=1)
print(json.dumps(out, indent=1))
