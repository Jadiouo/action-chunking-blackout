"""Fit the generic frame-anomaly detector on CLEAN calibration seeds only (python3 with numpy).
Input : .scratch/p2/feat/seed{100..119}.npy  (per-step ACT-backbone layer4 features, adaptive-avg-pooled to 3x4 -> 6144-d, fp16;
        produced by `code/p2p3.sh calib`, fixed n=50, L=0, i.e. no corruption).
Fit   : seeds 100-109  -> mean, PCA (top K components of the covariance), per-component variance.
Thresh: seeds 110-119  -> score = PCA reconstruction error SPE = || (x-mu) - V V^T (x-mu) ||_2 (distance to the clean-frame subspace;
        a first version used the Mahalanobis distance inside the K-d subspace and did NOT detect anything (AUC ~ 0.5, scores of black/noisy
        frames fall INSIDE the clean range) -> replaced, see docs/p2p3_notes.md);
        threshold = 99th percentile of the per-step scores on the held-out clean episodes (target false-alarm rate 1 % per step).
Output: results/parts/generic_detector.npz (mu, V, lam, thr) and results/parts/generic_detector.json (summary)."""
import glob, json, sys
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
FIT, THR = range(100, 110), range(110, 120)
load = lambda seeds: np.concatenate([np.load(f".scratch/p2/feat/seed{s}.npy").astype(np.float32) for s in seeds])
Xf, Xt = load(FIT), load(THR)
mu = Xf.mean(0)
U, S, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
V = Vt[:K].T.astype(np.float32); lam = (S[:K] ** 2 / (len(Xf) - 1)).astype(np.float32)
score = lambda X: np.linalg.norm((X - mu) - ((X - mu) @ V) @ V.T, axis=1)
sf, st = score(Xf), score(Xt)
thr = float(np.percentile(st, 99))
np.savez("results/parts/generic_detector.npz", mu=mu, V=V, lam=lam, thr=thr)
info = dict(K=K, fit_frames=len(Xf), thr_frames=len(Xt), feat_dim=int(Xf.shape[1]), thr=thr,
            fit_score_p50=float(np.percentile(sf, 50)), fit_score_p99=float(np.percentile(sf, 99)),
            heldout_score_p50=float(np.percentile(st, 50)), heldout_score_p99=thr, heldout_score_max=float(st.max()),
            heldout_frac_above_thr=float((st > thr).mean()), explained_var_topK=float((S[:K] ** 2).sum() / (S ** 2).sum()))
json.dump(info, open("results/parts/generic_detector.json", "w"), indent=1)
print(json.dumps(info, indent=1))
