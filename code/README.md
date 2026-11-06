# ACT chunk-length x camera-blackout pilot
Policy: lerobot/act_aloha_sim_transfer_cube_human (public; this legacy checkpoint has no processor files, so
normalization uses the mean/std buffers inside model.safetensors, applied manually in run_grid.py).
Env: gym_aloha/AlohaTransferCube-v0, obs_type=pixels_agent_pos, 400 steps; success = max reward 4.

Versions: python 3.10, torch 2.10.0+cu128, lerobot 0.4.4, gym-aloha 0.1.5, mujoco 3.8.1, gymnasium 1.4.0, dm-control 1.0.41;
GPU NVIDIA GeForce RTX 5070 Ti (16GB), MUJOCO_GL=egl.

Execution steps n: policy.config.n_action_steps = n then policy.reset() (queue maxlen=n); chunk_size stays 100.
TE: temporal_ensemble_coeff=0.01, n_action_steps=1 (policy rebuilt with same weights).
Blackout (code/blackout.py): only the image given to the policy for steps [t0, t0+L); agent_pos untouched.
t0 ~ Uniform{50..250} from RNG seeded by (10000+seed0+episode), env seed = seed0+episode: identical across cells.
Replans counted as queue-empty events (every step for TE).

Mean seconds/episode: ~1.3-1.9 s (n>=10), ~3.4 s (n=1 / TE); overall mean in report; 900 episodes total.

Rerun (from project root, HF token env unset-safe):
  export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl
  gpujob systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 bash code/run_all.sh
  .venv/bin/python code/summarize.py

T1/T2 (2026-11-03): code/run_t1.sh, code/run_t2.sh <A|B|C|D>, code/run_diag.sh {traj,off}, code/run_noise_repeat.sh;
analysis: code/check_determinism.py, code/analyze_n1.py (python3 with matplotlib), code/merge_full_grid.py,
`.venv/bin/python code/summarize.py results/full_grid.csv results/full_grid_table.md`.
Full-grid jobs ran with OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 (machine CPU saturated). Note the pipeline is not bit-deterministic (see docs/rulers.md).

T3/T4 (2026-11-03): code/run_t3t4.py (own execution loop; rules fixed|oracle|proprio), code/run_t3t4.sh <job> (smoke, oracle <n>, calib_a|calib_b, proprio_L0, proprio_main <black|freeze>, sens_arm95, sens_all99),
analysis (python3): code/make_oracle_table.py -> results/oracle.csv, oracle_table.md; code/proprio_calib.py <arm|all> -> figs + results/parts/proprio_calib_stats_*.json (tau from seeds 100-119); code/make_proprio_tables.py -> results/proprio.csv, parts/proprio_tables.md.
Same env vars/threads as T2. proprio_signal.md narrative is hand-written from these tables.
T4b (2026-11-03): code/run_t3t4.sh detect_smoke|detect_calib|detect_main <25|50> (rule detect); code/make_detect_table.py -> results/detect.csv, detect_table.md (python3).
P2/P3 (2026-11-04): code/corrupt.py (noise|occlude|delay|flicker|black|freeze), code/run_p2p3.py (own loop; rules fixed|oracle|detect|generic|median_te|mean_te|care), code/p2p3.sh <job> (smoke, pilot, calib, p2_main <corr> <param> [mini], diag, care_check, care_check2, p3_te, p3_cmp, p3_noise_te, p3_care [chains p3_fixed], regress100, regress_grid; run via gpujob),
  calibration (python3): code/calib_generic.py [K] (needs .scratch/p2/feat from `p2p3.sh calib`) -> results/parts/generic_detector.npz; code/calib_care.py -> results/parts/care_tau_calib.json (tau actually used: results/parts/care_tau_used.json).
  tables (python3, PYTHONPATH=code): code/make_realistic.py -> results/realistic.csv, realistic_table.md; code/make_baselines.py -> results/baselines.csv, baselines_table.md. Notes: docs/p2p3_notes.md.
P1 Insertion (2026-11-04): code/run_insertion.sh <noblack|fixed_black|fixed_freeze|rerun|oracle|detect> (fixed rows via run_grid.py --task insertion, oracle/detect rows via run_t3t4.py --task insertion; 200 seeds),
  merge + tables (python3, from project root): code/make_insertion_table.py -> results/insertion.csv, insertion_table.md. Success = max reward 4 (code/tasks.py).
P2b (2026-11-04..06; pre-registration docs/p2b_prereg.md): code/p2b.sh <smoke|tc_clean|dtype|pilot|pilot_ext|unseen_main <blur|dim>|clean_cost|te|ins_clean>
  (same-loop wrapper code/run_p2b.py with code/corrupt_p2b.py; submit as in the header of code/p2b.sh, via gpujob with systemd-run memory cap),
  tables (python3, PYTHONPATH=code): python3 code/make_p2b.py select -> results/parts/p2b_selected.json; python3 code/make_p2b.py table -> results/p2b_*.csv, p2b_table.md. The OOD detector is frozen by sha256 (checked by run_p2b.py).
  reward-4 meaning: code/inspect_insertion.py (frames) and code/measure_insertion_depth.py (peg depth from the simulator) -> results/insertion_frames/, results/insertion_inspect.csv, results/insertion_depth.csv
Diffusion Policy (negative result): code/dp_train.sh, code/dp_run.sh <run> <steps> <batch> [gate_step], code/dp_eval.py, code/dp_grid.py; rationale in docs/dp_training.md.
P5 (2026-11-06): code/analyze_arxiv.py (python3, no GPU) -> results/arxiv_numbers.json, paper/arxiv/figs/*.pdf, paper/arxiv/tables/*.tex (reads realistic.csv with stage=='main' only);
  paper: bash paper/arxiv/build.sh ; numbers: python3 paper/arxiv/verify_numbers.py (exit 0 means no mismatch among the values it checks; Appendix A.6 of the paper says what is not checked).
  The paper points to https://github.com/Jadiouo/action-chunking-blackout, tag v1 (macros \repourl, \repocommit at the top of paper/arxiv/main.tex); that repository is a clean snapshot without the original git history.
