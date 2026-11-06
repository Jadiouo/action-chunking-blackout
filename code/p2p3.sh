#!/bin/bash
# P2/P3 jobs. Usage (project root, via gpujob):  bash code/p2p3.sh <job> [args]
# env: see README.md. All outputs go to results/parts/p2p3_* (merged by code/make_realistic.py / make_baselines.py) or .scratch/p2.
export HF_HOME=$PWD/.hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/bin/python
R="$PY code/run_p2p3.py"
P=results/parts
case "$1" in
  smoke) # regression vs results/full_grid.csv (fixed rule, same seeds) + one episode of every new rule
     $R --rule fixed --n 25 --L 0 --episodes 5 --out .scratch/p2/reg_fixed_L0.csv &&
     $R --rule fixed --n 25 50 --L 50 --corr black  --episodes 10 --out .scratch/p2/reg_fixed_black.csv &&
     $R --rule fixed --n 25 50 --L 50 --corr freeze --episodes 10 --out .scratch/p2/reg_fixed_freeze.csv &&
     $R --rule median_te --h 1 10 --L 50 --corr black --episodes 2 --out .scratch/p2/smoke_mte.csv &&
     $R --rule mean_te --h 1 --L 0 --episodes 5 --out .scratch/p2/smoke_mean.csv &&
     $R --rule care --tau 0.05 --L 50 --corr black --episodes 2 --out .scratch/p2/smoke_care.csv &&
     $R --rule oracle --n 25 --L 50 --corr black --episodes 10 --out .scratch/p2/reg_oracle_black.csv &&
     $R --rule detect --n 25 --L 50 --corr freeze --episodes 10 --out .scratch/p2/reg_detect_freeze.csv ;;
  regress100) # implementation-vs-run_grid.py flip rate in blind cells (fixed rule, seeds 0-99; compare with results/full_grid.csv)
     $R --rule fixed --n 25 50 --L 50 --corr black  --out $P/p2p3_regress_black.csv &&
     $R --rule fixed --n 25 50 --L 50 --corr freeze --out $P/p2p3_regress_freeze.csv ;;
  pilot) # P2 intensity pilot: fixed n=50, L=50, seeds 120-149 (disjoint from eval 0-99 and calibration 100-119), 30 episodes
     S="--seed0 120 --episodes 30"
     $R --rule fixed --n 50 --L 0 $S --out $P/p2p3_pilot.csv &&
     for p in 0.1 0.3 0.6; do $R --rule fixed --n 50 --L 50 --corr noise --param $p $S --out $P/p2p3_pilot.csv; done &&
     for p in 0.25:center 0.5:center 0.75:center 0.25:rand 0.5:rand 0.75:rand; do $R --rule fixed --n 50 --L 50 --corr occlude --param $p $S --out $P/p2p3_pilot.csv; done &&
     for p in 5 25 50; do $R --rule fixed --n 50 --L 50 --corr delay --param $p $S --out $P/p2p3_pilot.csv; done &&
     for p in 0.3 0.6 1.0; do $R --rule fixed --n 50 --L 50 --corr flicker --param $p $S --out $P/p2p3_pilot.csv; done ;;
  calib) # clean calibration seeds 100-119: backbone features (generic detector) and CARE discrepancy curves (open loop, n=30)
     $R --rule fixed --n 50 --L 0 --seed0 100 --episodes 20 --feat_dir .scratch/p2/feat --out .scratch/p2/calib_feat.csv &&
     $R --rule fixed --n 30 --L 0 --seed0 100 --episodes 20 --curve_dir .scratch/p2/curve --out .scratch/p2/calib_curve.csv ;;
  p3_te) # Median TE (h=1 main, h=10 secondary) on Transfer Cube black / freeze, L in {25,50,100} + clean
     $R --rule median_te --h 1 10 --L 0 25 50 100 --corr black  --out $P/p2p3_p3_median_black.csv &&
     $R --rule median_te --h 1 10 --L 25 50 100 --corr freeze --out $P/p2p3_p3_median_freeze.csv ;;
  p3_cmp) # extra comparators on black/freeze: oracle & detect n=100 and L=25 (n=25,50 detect; n=100 oracle/detect), L=50/100 n=100
     for m in black freeze; do
       $R --rule oracle --n 100 --L 25 50 100 --corr $m --out $P/p2p3_p3_oracle_$m.csv &&
       $R --rule detect --n 25 50 --L 25 --corr $m --out $P/p2p3_p3_detect_$m.csv &&
       $R --rule detect --n 100 --L 25 50 100 --corr $m --out $P/p2p3_p3_detect_${m}_n100.csv || exit 1
     done ;;
  care_check) # closed-loop mean H of CARE on clean seeds 100-119 for the open-loop-calibrated taus (results/parts/care_tau_calib.json)
     for t in 0.1069 0.1748 0.3544; do $R --rule care --tau $t --L 0 --seed0 100 --episodes 20 --out $P/p2p3_carecheck.csv || exit 1; done ;;
  care_check2) # second closed-loop calibration round (open-loop estimate was too low: realised mean H 13/19/33 for targets 20/30/40)
     for t in 0.18 0.30 0.50 0.80; do $R --rule care --tau $t --L 0 --seed0 100 --episodes 20 --out $P/p2p3_carecheck.csv || exit 1; done ;;
  regress_grid) # run-to-run flip rate of the ORIGINAL pipeline (run_grid.py) in a blind cell, to compare with the flips of regress100
     $PY code/run_grid.py --n 25 --L 50 --mode freeze --episodes 100 --out $P/p2p3_regress_rungrid_freeze.csv ;;
  p3_noise_te) # P3 on the P2 corruption noise sigma=0.6 (90 seeds): Median TE h=1,10 and Mean TE h=1 (no full_grid reference exists under noise)
     $R --rule median_te --h 1 10 --L 50 100 --corr noise --param 0.6 --episodes 90 --out $P/p2p3_p3_noise_te.csv &&
     $R --rule mean_te --h 1 --L 50 100 --corr noise --param 0.6 --episodes 90 --out $P/p2p3_p3_noise_te.csv ;;
  p3_care) # CARE: main tau=0.30 (closed-loop clean mean H 31.5 ~ 0.6*H_max) on black/freeze L in {25,50,100} + clean; low 0.18 (H~21) / high 0.50 (H~38) at L in {0,50}
     for t in 0.30; do
       $R --rule care --tau $t --L 0 25 50 100 --corr black  --out $P/p2p3_p3_care_black.csv &&
       $R --rule care --tau $t --L 25 50 100 --corr freeze --out $P/p2p3_p3_care_freeze.csv || exit 1
     done
     for t in 0.18 0.50; do
       $R --rule care --tau $t --L 0 50 --corr black  --out $P/p2p3_p3_care_black.csv &&
       $R --rule care --tau $t --L 50 --corr freeze --out $P/p2p3_p3_care_freeze.csv || exit 1
     done
     $R --rule care --tau 0.30 --L 50 100 --corr noise --param 0.6 --episodes 90 --out $P/p2p3_p3_care_noise.csv &&
     bash "$0" p3_fixed ;;  # (p3_fixed is chained here so that it is not queued behind other projects' long jobs)
  p3_fixed) # same-loop (float64, run_p2p3.py) fixed-n and Mean TE rows for black/freeze L in {25,50,100}: the run_grid.py rows in full_grid.csv
            # differ per episode in ~6-14% of mid-range blind cells (float32 actions -> chaotic grasps), so P3 pairs the methods with same-loop rows.
     for m in black freeze; do
       $R --rule fixed --n 25 50 100 --L 25 100 --corr $m --out $P/p2p3_p3_fixed_$m.csv &&
       $R --rule fixed --n 100 --L 50 --corr $m --out $P/p2p3_p3_fixed_$m.csv &&
       $R --rule mean_te --h 1 --L 25 50 100 --corr $m --out $P/p2p3_p3_meante_$m.csv || exit 1
     done ;;
  diag) # harm-label diagnostics: fixed n=50, L=50, seeds 120-149, 3 forward passes/step (needs results/parts/generic_detector.npz)
     D="$R --rule fixed --n 50 --L 50 --seed0 120 --episodes 30"
     $D --corr noise   --param 0.6        --diag_dir $P/p2p3_diag_noise0.6        --out .scratch/p2/diag_noise.csv &&
     $D --corr occlude --param 0.5:center --diag_dir $P/p2p3_diag_occlude0.5center --out .scratch/p2/diag_occ.csv &&
     $D --corr delay   --param 50         --diag_dir $P/p2p3_diag_delay50         --out .scratch/p2/diag_delay.csv &&
     $D --corr flicker --param 0.6        --diag_dir $P/p2p3_diag_flicker0.6      --out .scratch/p2/diag_flicker.csv &&
     $D --corr black                      --diag_dir $P/p2p3_diag_black           --out .scratch/p2/diag_black.csv &&
     $D --corr freeze                     --diag_dir $P/p2p3_diag_freeze          --out .scratch/p2/diag_freeze.csv ;;
  p2_main) # usage: p2_main <corr> <param> [mini]; 90 seeds. full: fixed/oracle/generic n{25,50,100} x L{50,100} + simple-detect n=50; mini: n=50 only, no detect
     c=$2; q=$3
     if [ "$4" = mini ]; then NS="50"; else NS="25 50 100"; fi
     for r in fixed oracle generic; do $R --rule $r --n $NS --L 50 100 --corr $c --param $q --episodes 90 --out $P/p2p3_real_${c}_${q}.csv || exit 1; done
     if [ "$4" != mini ]; then $R --rule detect --n 50 --L 50 100 --corr $c --param $q --episodes 90 --out $P/p2p3_real_${c}_${q}.csv; fi ;;
  *) echo "unknown job $1"; exit 1 ;;
esac
