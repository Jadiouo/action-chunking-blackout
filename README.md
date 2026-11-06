# Action chunking under camera blackout

Code, result tables and checks for the paper *Action Chunking under Camera Blackout: Replanning on Bad Images as a Failure Mechanism* (Hao-Ting Ho, 2026). The source is `paper/arxiv/main.tex`.

The study asks what the execution horizon `n` of an action-chunking policy does when the camera fails for `L` steps while the joint readings stay valid. It uses pretrained ACT policies on simulated ALOHA Transfer Cube and Insertion. Everything is simulated.

This repository is a clean snapshot, tagged `v1`. It has no git history from the author's earlier work. The date of the pre-registration (`docs/p2b_prereg.md`) therefore cannot be checked from this repository. Its sha256 is `6d5460847cf0c0feb74ef3486aebed88033193149e1b2ba446efee203c10ca60`; the frozen detector `results/parts/generic_detector.npz` has sha256 `5193a96b546ffd97d4632bf2c1eb8b01b40b049748c77e62241cf600e244c2f2` (also written in the pre-registration and checked by `code/run_p2b.py`). The original history can be given to reviewers on request.

## What is here

- `code/`: simulation loops, analysis scripts, job scripts. `code/README.md` gives the commands for the main scripts, in the order they were run.
- `results/`: the per-episode result files (`*.csv`), the tables derived from them, and `results/parts/` with the raw parts, the detector calibration and the frozen detector.
- `paper/arxiv/`: `main.tex`, `refs.bib`, generated tables and figures, `build.sh`, and `verify_numbers.py`.
- `docs/`: protocol notes (`rulers.md`, `baselines_spec.md`, `p2p3_notes.md`, `dp_training.md`), the pre-registration (`p2b_prereg.md`) and the literature record (`literature.md`). Notes are mostly in Chinese.

## Environment

- Analysis only (no GPU, no simulator): `requirements-analysis.txt` (Python 3.12; numpy, pandas, scipy, matplotlib), plus a TeX Live with `latexmk`, `pdflatex` and `bibtex` for the paper.
- Simulation: create a virtual environment at `.venv` in the repository root, because the job scripts call `.venv/bin/python`, and install `requirements-sim.txt` into it. Python 3.10, PyTorch 2.10.0 (CUDA 12.8), LeRobot 0.4.4, gym-aloha 0.1.5, MuJoCo 3.8.1, EGL rendering, one NVIDIA RTX 5070 Ti.

## Reproduce

Run everything from the repository root.

```bash
# 1. Tables, figures and results/arxiv_numbers.json from the CSV files (a few seconds, CPU)
python3 code/analyze_arxiv.py

# 2. Check the reported numbers against the CSV files and the notes (exit 0 = no mismatch found; Appendix A.6 of the paper says what is not checked)
python3 paper/arxiv/verify_numbers.py

# 3. Build the paper (writes paper/arxiv/main.pdf)
bash paper/arxiv/build.sh
```

`verify_numbers.py` checks the numbers and a list of key sentences. It also checks the order of the pre-registration and the runs when the original git history is present. In this snapshot it skips that part and prints a note.

The simulation runs (hours of GPU time) are only needed to regenerate the CSV files. The commands are in `code/README.md`; for example the first pilot grid is `bash code/run_all.sh`, the full Transfer Cube grid uses `code/run_t1.sh`, `code/run_t2.sh` and `code/merge_full_grid.py`, and the later experiments use `code/run_t3t4.sh`, `code/p2p3.sh`, `code/run_insertion.sh` and `code/p2b.sh`. Set the environment first:

```bash
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MUJOCO_GL=egl
```

The scripts write scratch files to `.scratch/` and model files to `.hf/` and `runs/`; all three are ignored by git. Several job scripts and docs mention `gpujob`. That is the author's own local GPU queue wrapper and is optional: drop the `gpujob` prefix and run the command directly. The `systemd-run --user --scope -p MemoryMax=12G ...` prefix is a memory cap, also optional.

The pipeline is not bit-deterministic (see `docs/rulers.md` and Appendix A.3 of the paper): rerunning a simulation can flip a few episodes, so a rerun will not reproduce the CSV files exactly. Marginal success rates are what the paper claims.

## Data and weights

No weights, datasets or simulator code are included. The policies are the public Hugging Face checkpoints `lerobot/act_aloha_sim_transfer_cube_human` and `lerobot/act_aloha_sim_insertion_human`, downloaded at run time. The files in `results/` are the author's own outputs.

## Licenses of what this work uses

- This repository: MIT, see `LICENSE`.
- `lerobot/act_aloha_sim_transfer_cube_human` and `lerobot/act_aloha_sim_insertion_human`: the Hugging Face model cards declare `apache-2.0` (checked 2026-11-06).
- gym-aloha: Apache-2.0 (GitHub repository page, and the package metadata of 0.1.5). `docs/gym_aloha_excerpt.txt` holds a few lines of its source so the number check can run without installing it.
- LeRobot: Apache-2.0 (package metadata of 0.4.4).

## Use of generative AI

The experiment code, the analysis and the text of this paper were largely written by Claude (Anthropic), an AI assistant, under the author's direction and review. The author takes responsibility for the content.
