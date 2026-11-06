"""P2b tables (docs/p2b_prereg.md).  python3 code/make_p2b.py select   -> results/parts/p2b_selected.json (intensity rule of the pilot)
                                    python3 code/make_p2b.py table    -> results/p2b_*.csv (merged raw rows), results/p2b_table.md
Every number in p2b_table.md is computed here from the CSVs.  Pairing key = env_seed (same t0 in every cell; asserted).
Holm only inside the pre-registered families F1 (Insertion), F2 (unseen corruptions), F3 (Median TE); everything else is exploratory
(unadjusted p).  Differences |delta| < 0.05 are not interpreted (docs/rulers.md)."""
import glob, json, os, sys
import numpy as np, pandas as pd
from p2p3_stats import rate_ci, mcnemar, holm, fmt_R, fmt_rho, wilson, recovery, RERUN_NOISE

P = "results/parts/"
DET_SHA = "5193a96b546ffd97d4632bf2c1eb8b01b40b049748c77e62241cf600e244c2f2"
CORR_ORDER = {"blur": [5, 15, 40, 80], "dim": [0.5, 0.25, 0.1, 0.03]}  # weak -> strong
EXT = {"blur": 80, "dim": 0.03}


def g_(x):
    return f"{x:g}"


def rd(f):
    return pd.read_csv(f) if os.path.exists(f) else None


def vec(df):
    return df.sort_values("env_seed")


def paired(a, b):
    a, b = vec(a), vec(b)
    assert (a.env_seed.values == b.env_seed.values).all() and (a.t0.values == b.t0.values).all(), "pairing mismatch"
    return a, b


def boot_diff(a, b, B=4000, seed=3):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(B, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fd(a, b):
    d, lo, hi = boot_diff(a, b)
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


# ------------------------------------------------------------------------------------------------ select
def select():
    pil = rd(P + "p2b_pilot.csv")
    if pil is None:
        sys.exit("no pilot")
    clean = vec(pil[pil.L == 0]).success.values
    assert len(clean) == 30
    out, need = {"pilot_clean": f"{int(clean.sum())}/30", "R": {}}, []
    for c, order in CORR_ORDER.items():
        rows = {}
        for s in order:
            g = pil[(pil.L == 50) & (pil["corr"] == f"{c}{g_(s)}")]
            if len(g): rows[s] = g.success.mean() / clean.mean()
        out["R"][c] = {g_(k): round(v, 3) for k, v in rows.items()}
        base = [s for s in order[:3] if s in rows]
        assert len(base) == 3, f"pilot incomplete for {c}"
        cand = {s: r for s, r in rows.items() if 0.15 <= r <= 0.85}
        if not cand:
            if all(rows[s] > 0.85 for s in base) and EXT[c] not in rows:
                need.append(c); continue
            if all(r > 0.85 for r in rows.values()):
                ch, why = order[max(i for i, s in enumerate(order) if s in rows)], "all R>0.85: strongest, near-harmless"
            else:
                ch, why = [s for s in order if s in rows][0], "all R<0.15: weakest"
        else:
            best = min(abs(r - 0.5) for r in cand.values())
            ch = [s for s in order if s in cand and abs(cand[s] - 0.5) == best][0]  # order is weak -> strong: ties go to the weaker
            why = "closest R to 0.5 within [0.15,0.85]"
        out[c] = g_(ch); out.setdefault("why", {})[c] = why
    out["need_ext"] = need
    json.dump(out, open(P + "p2b_selected.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    if need:
        print("NEED pre-declared 4th intensity:", {c: EXT[c] for c in need})
        sys.exit(2)


# ------------------------------------------------------------------------------------------------ table
def table():
    out = []
    W = out.append
    verdict = []
    V = verdict.append
    allcsv = sorted(glob.glob(P + "p2b_*.csv"))
    gpu_h = sum(pd.read_csv(f).seconds.sum() for f in allcsv) / 3600
    nrows = {os.path.basename(f): len(pd.read_csv(f)) for f in allcsv}
    W("# P2b 補做：同迴圈對照、dtype 消融、未見損壞、Median TE 確認重跑\n")
    W("本檔由 `code/make_p2b.py` 從 `results/parts/p2b_*.csv`（合併檔 `results/p2b_*.csv`）產生；預測、強度規則、主對比家族在 `docs/p2b_prereg.md`（先於任何 GPU 工作 commit）。"
      "程式：`code/run_p2b.py`（`run_p2p3.run_episode` 原迴圈）、`code/corrupt_p2b.py`、`code/p2b.sh`。"
      f"偵測器凍結（sha256 `{DET_SHA[:12]}…`，由 `run_p2b.py` 啟動時檢查）。Wilson 95% CI；McNemar 精確；Holm 只在預先登記家族 F1／F2／F3 內，其餘為探索性（未校正 p）；"
      f"R、ρ 的方括號為 seed 配對 bootstrap 95% CI（4000 次）；ρ 分母 <0.10 記 n/r；差距 <{RERUN_NOISE} 標 noise 不解讀。\n")
    W(f"GPU：本批 episode 秒數加總 **{gpu_h:.2f} h**（預算 ≤3.5 h；煙霧測試在 `.scratch/p2b`，另計約 0.01 h）。列數：" + "，".join(f"{k} {v}" for k, v in nrows.items()) + "。\n")

    # ===================================================================================== #1.2 dtype ablation and TC clean
    grid = pd.read_csv("results/full_grid.csv")
    W("## 1. 事項 #1b：dtype 消融與 Transfer Cube 乾淨參照\n")
    dt = rd(P + "p2b_dtype.csv")
    if dt is not None:
        W("### 1.1 dtype 消融（TC，n=25、L=50、seeds 0–99）\n")
        W("對照三條路徑：`run_grid.py`（policy 佇列，動作 float32，`full_grid.csv`）、新迴圈 float64（`p2p3_regress_*`，P2 當時的結果）、新迴圈 float32（本次）。"
          "「翻轉」＝逐 episode 成敗與 `run_grid.py` 列不同的數量。\n")
        W("| 模式 | run_grid（f32，舊迴圈） | 新迴圈 f64 | 新迴圈 f32 | f64 對 run_grid 翻轉 | **f32 對 run_grid 翻轉** | f32 對 f64 翻轉 |"); W("|---|---|---|---|---|---|---|")
        tot32 = tot64 = 0; ncell = 0
        for m in ("black", "freeze"):
            old = grid[(grid["mode"] == m) & (grid.n == 25) & (grid.L == 50) & (grid.te == 0) & (grid.episode < 100)]
            f64 = pd.read_csv(P + f"p2p3_regress_{m}.csv"); f64 = f64[(f64.n == 25) & (f64.L == 50)]
            f32 = dt[(dt["corr"] == m) & (dt.n == 25) & (dt.L == 50)]
            if len(f32) < 100: continue
            ncell += 1
            o_, a64 = vec(old), vec(f64); a32 = vec(f32)
            assert (o_.env_seed.values == a32.env_seed.values).all() and (a64.env_seed.values == a32.env_seed.values).all()
            fl64 = int((o_.success.values != a64.success.values).sum()); fl32 = int((o_.success.values != a32.success.values).sum())
            f3264 = int((a32.success.values != a64.success.values).sum()); tot64 += fl64; tot32 += fl32
            W(f"| {m} | {rate_ci(o_.success.values)} | {rate_ci(a64.success.values)} | {rate_ci(a32.success.values)} | {fl64}/100 | **{fl32}/100** | {f3264}/100 |")
        rr = rd(P + "p2p3_regress_rungrid_freeze.csv")
        if rr is not None:
            old = vec(grid[(grid["mode"] == "freeze") & (grid.n == 25) & (grid.L == 50) & (grid.te == 0) & (grid.episode < 100)])
            W(f"\n雜訊底線：`run_grid.py` 自己重跑 freeze n=25 L=50 的翻轉 {int((vec(rr).success.values != old.success.values).sum())}/100。\n")
        if ncell == 2:
            if tot32 <= 8: vd = f"成立（{tot32}/200 ≤ 8，f64 為 {tot64}/200）：dtype 是主因，已驗證"
            elif tot32 >= 14: vd = f"不成立（{tot32}/200 ≥ 14，f64 為 {tot64}/200）：dtype 不是主因（或非唯一原因），寫成未解"
            else: vd = f"部分（{tot32}/200 介於 9–13，f64 為 {tot64}/200）：部分由 dtype 解釋"
            W(f"預先登記判準（合計翻轉 ≤8/200＝主因；≥14/200＝非主因）：**{vd}**\n")
            V(f"dtype 消融預測（f32 翻轉 ≤8/200）：{vd}")
    cl = rd(P + "p2b_tc_clean_fixed.csv")
    if cl is not None:
        W("### 1.2 TC 同迴圈乾淨參照（L=0，seeds 0–89）\n")
        W("| n | 新迴圈 fixed（f64） | `full_grid.csv`（run_grid） | Δ | 翻轉 |"); W("|---|---|---|---|---|")
        mx = 0
        for n in (25, 50, 100):
            a = vec(cl[cl.n == n]); o_ = vec(grid[(grid.n == n) & (grid.L == 0) & (grid.te == 0) & (grid.episode < 90)])
            if len(a) < 90: continue
            assert (a.env_seed.values == o_.env_seed.values).all()
            d = a.success.mean() - o_.success.mean(); mx = max(mx, abs(d))
            W(f"| {n} | {rate_ci(a.success.values)} | {rate_ci(o_.success.values)} | {d:+.2f} | {int((a.success.values != o_.success.values).sum())}/90 |")
        W("")
        V(f"TC 乾淨參照預測（每 n 與 full_grid 相差 ≤0.05）：{'成立' if mx <= 0.05 else '不成立'}（最大 |Δ|={mx:.3f}）")

    # ===================================================================================== #1.1 Insertion
    fl = [rd(P + f) for f in ("p2b_ins_fixed_L50.csv", "p2b_ins_fixed_L100.csv", "p2b_ins_fixed_L0.csv")]
    fl = [x for x in fl if x is not None]
    if fl:
        new = pd.concat(fl, ignore_index=True)
        old = pd.read_csv("results/insertion.csv")
        oldf = old[old.rule == "fixed"]
        pd.concat([new], ignore_index=True).to_csv("results/p2b_insertion_fixed.csv", index=False)
        cleann = {n: new[(new.L == 0) & (new.n == n)] for n in (25, 50, 100)}

        def one(metric, title, tag_, famflag):
            W(f"### {title}\n")
            W("成功＝" + ("`max_reward ≥ 4`（預先宣告主指標）" if metric == 4 else "`max_reward ≥ 3`（**事後、次要**指標：兩物件離桌且接觸；看過主指標功效不足後才加）") + "。"
              "new fixed＝本次同迴圈（float64）；old fixed＝`run_grid.py`（float32）；oracle／detect 為 `insertion.csv`（`run_t3t4.py`，float64，與新迴圈同路徑）。"
              "Δ、p 均相對 **new fixed**；p 欄 `F1:` 為 Holm 後（預先登記家族），其餘為未校正（探索性）；R、ρ 的乾淨參照＝new L=0 同 n。\n")
            sc = (lambda d: (d.max_reward >= metric).astype(int))
            W("| 模式 | L | n | new clean | old fixed | **new fixed** | 翻轉 | oracle | detect |"); W("|---|---|---|---|---|---|---|---|---|")
            fam = []  # (key, x, y, p)
            rows = []
            for m in ("black", "freeze"):
                for L in (50, 100):
                    for n in (25, 50, 100):
                        nf = new[(new["corr"] == m) & (new.L == L) & (new.n == n)]
                        if len(nf) < 200: continue
                        of = oldf[(oldf["mode"] == m) & (oldf.L == L) & (oldf.n == n)]
                        cn = cleann[n]
                        cell = {}
                        a, o_ = paired(nf, of)
                        cell["flip"] = int((sc(a).values != sc(o_).values).sum())
                        for r in ("oracle", "detect"):
                            rr = old[(old.rule == r) & (old["mode"] == m) & (old.L == L) & (old.n == n)]
                            if len(rr) == 200:
                                x_, y_ = paired(rr, nf)
                                x, y, p = mcnemar(sc(x_).values, sc(y_).values)
                                cell[r] = (x_, y_, x, y, p)
                                fam.append(((m, L, n, r), x, y, p))
                        rows.append((m, L, n, nf, of, cn, cell))
            fam_keys = [(("black", 50, 25, "oracle")), (("black", 50, 25, "detect")), (("black", 50, 50, "oracle"))]
            fam_p = [next(v[3] for v in fam if v[0] == k) for k in fam_keys if any(v[0] == k for v in fam)]
            fam_adj = dict(zip([k for k in fam_keys if any(v[0] == k for v in fam)], holm(fam_p))) if (famflag and fam_p) else {}
            for m, L, n, nf, of, cn, cell in rows:
                a = sc(vec(nf)).values; clean = sc(vec(cn)).values if len(cn) == 200 else None
                cs = [rate_ci(clean) if clean is not None else "-", rate_ci(sc(vec(of)).values), rate_ci(a), f"{cell['flip']}/200"]
                for r in ("oracle", "detect"):
                    if r not in cell: cs.append("-"); continue
                    x_, y_, x, y, p = cell[r]
                    v = sc(x_).values; d = v.mean() - a.mean()
                    rho = fmt_rho(v, a, clean) if clean is not None else "n/a"
                    Rr = fmt_R(v, clean) if clean is not None else "n/a"
                    key = (m, L, n, r)
                    ptxt = f"F1:{fam_adj[key]:.3g}" if key in fam_adj else f"p={p:.3g}"
                    cs.append(f"{rate_ci(v)}<br>R={Rr}; ρ={rho}<br>Δ={d:+.2f}{' noise' if abs(d) < RERUN_NOISE else ''}; {x}/{y}; {ptxt}")
                W(f"| {m} | {L} | {n} | " + " | ".join(cs) + " |")
            W("")
            return fam_adj, fam

        fam_adj, fam = one(4, "2.1 Insertion：同迴圈 fixed 對 oracle／detect（reward=4，主指標）", "r4", True)
        # F1 verdict
        old_sig = {}
        for k in [("black", 50, 25, "oracle"), ("black", 50, 25, "detect"), ("black", 50, 50, "oracle")]:
            if k in fam_adj:
                v = next(v for v in fam if v[0] == k)
                old_sig[k] = (v[1], v[2], fam_adj[k])
        if len(fam_adj) == 3:
            ok = all(x > y and pa < 0.05 for x, y, pa in old_sig.values())
            W("**F1（Holm，3 個，black L=50，對 new fixed）**：" + "；".join(f"{k[3]} n={k[2]}：{x}/{y}，p_Holm={pa:.3g}" for k, (x, y, pa) in old_sig.items()) + f"。→ 結論不變判準：**{'成立' if ok else '不成立（見上，邊界格降級為「未確認」）'}**\n")
            V(f"Insertion F1（三項方向同舊且 Holm 後 p<0.05）：{'成立' if ok else '不成立'}（" + "；".join(f"{k[3]} n={k[2]} p_Holm={pa:.3g}" for k, (x, y, pa) in old_sig.items()) + "）")
        # marginal reproducibility (a)
        dm = []
        for m in ("black", "freeze"):
            for L in (50, 100):
                for n in (25, 50, 100):
                    nf = new[(new["corr"] == m) & (new.L == L) & (new.n == n)]; of = oldf[(oldf["mode"] == m) & (oldf.L == L) & (oldf.n == n)]
                    if len(nf) == 200: dm.append(((nf.max_reward >= 4).mean() - (of.max_reward >= 4).mean(), (m, L, n)))
        if dm:
            mx = max(dm, key=lambda t: abs(t[0]))
            V(f"Insertion 預測(a)（new fixed 與 old fixed 每格邊際相差 ≤0.05）：{'成立' if abs(mx[0]) <= 0.05 else '不成立'}（{len(dm)} 格，最大 |Δ|={abs(mx[0]):.3f} @ {mx[1]}）")
        # freeze L=50 oracle/detect |delta| < 0.05
        fz = [(abs((vec(old[(old.rule == r) & (old["mode"] == "freeze") & (old.L == 50) & (old.n == n)]).max_reward >= 4).mean() - (new[(new["corr"] == "freeze") & (new.L == 50) & (new.n == n)].max_reward >= 4).mean()), r, n)
              for r in ("oracle", "detect") for n in (25, 50) if len(new[(new["corr"] == "freeze") & (new.L == 50) & (new.n == n)]) == 200]
        if fz:
            V(f"Insertion 預測(c) freeze L=50 oracle／detect 對 new fixed |Δ|<0.05：{'成立' if max(f[0] for f in fz) < 0.05 else '不成立'}（最大 |Δ|={max(f[0] for f in fz):.3f}）")
        one(3, "2.2 同上，事後次要指標 reward ≥ 3（探索性，不做 Holm）", "r3", False)

    # ===================================================================================== #2 unseen corruptions
    sel = json.load(open(P + "p2b_selected.json")) if os.path.exists(P + "p2b_selected.json") else None
    pil = rd(P + "p2b_pilot.csv")
    if pil is not None:
        W("## 3. 事項 #2：未見過的損壞（偵測器凍結）\n")
        W("### 3.1 強度試跑（fixed n=50、L=50、seeds 120–149 共 30 個；乾淨參照＝同 seeds 同迴圈 L=0）\n")
        W("規則（預先登記）：R 最接近 0.5 且在 [0.15, 0.85]，平手取較弱者；三個 R 全 >0.85 才加預先宣告的第 4 強度（blur σ=80、dim g=0.03）。\n")
        pc = vec(pil[pil.L == 0]).success.values
        W("| 損壞 | 強度 | fixed 成功 | R | 入選 |"); W("|---|---|---|---|---|")
        W(f"| 乾淨 | - | {rate_ci(pc)} | 1.00 | |")
        for c, order in CORR_ORDER.items():
            for s in order:
                g = pil[(pil.L == 50) & (pil["corr"] == f"{c}{g_(s)}")]
                if len(g):
                    W(f"| {c} | {g_(s)} | {rate_ci(g.success.values)} | {g.success.mean() / pc.mean():.2f} | {'**入選**' if sel and sel.get(c) == g_(s) else ''} |")
        W("")
    for c in ("blur", "dim"):
        f = rd(P + f"p2b_unseen_{c}.csv")
        if f is None or sel is None: continue
        q = sel[c]; tg = f"{c}{q}"
        cleanf = rd(P + "p2b_tc_clean_fixed.csv")
        clean = vec(cleanf[cleanf.n == 50]) if cleanf is not None else None
        S = {r: vec(f[f.rule == r]) for r in ("fixed", "oracle", "generic")}
        if any(len(v) < 90 for v in S.values()): continue
        fx, orc, gen = S["fixed"], S["oracle"], S["generic"]
        paired(fx, orc); paired(fx, gen)
        W(f"### 3.2 {c}（入選強度 {q}，n=50、L=50、seeds 0–89）\n")
        cs = clean.success.values if clean is not None and len(clean) == 90 else None
        W("| 規則 | 成功 | R | ρ（對 fixed） | Δ 對 fixed [paired boot] | McNemar x/y | p |"); W("|---|---|---|---|---|---|---|")
        W(f"| 乾淨 fixed n=50 | {rate_ci(cs) if cs is not None else '-'} | | | | | |")
        for r in ("fixed", "oracle", "generic"):
            v = S[r].success.values
            if r == "fixed":
                W(f"| fixed | {rate_ci(v)} | {fmt_R(v, cs) if cs is not None else '-'} | | | | |"); continue
            x, y, p = mcnemar(v, fx.success.values)
            W(f"| {r} | {rate_ci(v)} | {fmt_R(v, cs) if cs is not None else '-'} | {fmt_rho(v, fx.success.values, cs) if cs is not None else '-'} | {fd(v, fx.success.values)} | {x}/{y} | "
              f"{'F2 見下' if r == 'generic' else f'{p:.3g}（探索）'} |")
        x, y, p = mcnemar(gen.success.values, orc.success.values)
        W(f"\ngeneric 對 oracle（探索性）：{x}/{y}，p={p:.3g}。\n")
        # detector metrics
        ed = (gen.det_in > 0).mean(); fin = gen.det_in.sum() / max(gen.steps_in.sum(), 1)
        ef = (gen.det_out > 0).mean(); fout = gen.det_out.sum() / max(gen.steps_out.sum(), 1)
        lat = pd.to_numeric(gen.latency, errors="coerce")
        W(f"偵測器指標：窗內被標記步數比例 **{fin:.3f}**；窗內至少偵測一次的 episode 比例 {ed:.2f}；窗外誤報率（步）**{fout:.4f}**；至少一次窗外誤報的 episode 比例 {ef:.2f}；"
          f"偵測延遲平均 {'-' if lat.notna().sum() == 0 else f'{lat.mean():.1f}'} 步；窗內被迫重規劃平均 {gen.forced_replans_in_window.mean():.2f} 次。\n")
        den = (cs.mean() - fx.success.mean()) if cs is not None else float("nan")
        rho = recovery(gen.success.values, fx.success.values, cs) if cs is not None else None
        rhov = rho[0] if rho else None
        thr = {"dim": (0.90, 0.60), "blur": (0.50, 0.40)}[c]
        ok1 = fin >= thr[0]; ok2 = (rhov is not None and rhov >= thr[1]) if rho else None
        V(f"{c} 預測（窗內標記比例 ≥{thr[0]}）：{'成立' if ok1 else '不成立'}（{fin:.3f}）；（ρ_generic ≥{thr[1]}）："
          + ("分母 <0.10，n/r" if rho is None else f"{'成立' if ok2 else '不成立'}（ρ={rhov:.2f}）") + f"；窗外誤報 ≤0.05：{'成立' if fout <= 0.05 else '不成立'}（{fout:.4f}）")
        notgen = (fin < 0.30) or (rho is not None and rhov < 0.30)
        V(f"{c}「偵測器不通用」判準（窗內標記 <0.30 或 ρ<0.30）：{'**觸發**' if notgen else '未觸發'}")
    # F2
    if sel is not None:
        fam = []
        for c in ("blur", "dim"):
            f = rd(P + f"p2b_unseen_{c}.csv")
            if f is None or len(f) < 270: continue
            a, b = paired(vec(f[f.rule == "generic"]), vec(f[f.rule == "fixed"]))
            x, y, p = mcnemar(a.success.values, b.success.values); fam.append((c, x, y, p))
        if fam:
            adj = holm([v[3] for v in fam])
            W("**F2（Holm，2 個）generic n=50 對 fixed n=50（L=50）**：" + "；".join(f"{c}：{x}/{y}，p={p:.3g}，p_Holm={a:.3g}" for (c, x, y, p), a in zip(fam, adj)) + "\n")
    gc = rd(P + "p2b_generic_clean.csv"); cl = rd(P + "p2b_tc_clean_fixed.csv")
    if gc is not None and cl is not None:
        W("### 3.3 乾淨代價：generic-detect L=0 對同迴圈 fixed L=0（TC，seeds 0–89；探索性 McNemar）\n")
        W("| n | fixed | generic-detect | Δ [paired boot] | x/y | p | 窗外（全程）誤報率（步） | 至少一次誤報的 episode |"); W("|---|---|---|---|---|---|---|---|")
        mx = 0; okn = 0
        for n in (25, 50, 100):
            a, b = vec(gc[gc.n == n]), vec(cl[cl.n == n])
            if len(a) < 90 or len(b) < 90: continue
            paired(a, b)
            x, y, p = mcnemar(a.success.values, b.success.values); d = a.success.mean() - b.success.mean(); mx = max(mx, abs(d)); okn += 1
            W(f"| {n} | {rate_ci(b.success.values)} | {rate_ci(a.success.values)} | {fd(a.success.values, b.success.values)} | {x}/{y} | {p:.3g} | "
              f"{a.det_out.sum() / max(a.steps_out.sum(), 1):.4f} | {(a.det_out > 0).mean():.2f} |")
        W("")
        if okn == 3:
            V(f"乾淨代價預測（每 n |Δ|<0.05）：{'成立' if mx < 0.05 else '不成立'}（最大 |Δ|={mx:.3f}）")

    # ===================================================================================== #3 TE
    te = rd(P + "p2b_te.csv")
    if te is not None:
        te["rname"] = te.apply(lambda r: {"median_te": f"Median TE h={r.h}", "fixed": f"fixed n={r.n}", "oracle": f"oracle n={r.n}"}[r.rule], axis=1)
        te.to_csv("results/p2b_te.csv", index=False)
        W("## 4. 事項 #3：Median TE 確認性重跑（TC，新 seeds 150–249，100 個）\n")
        cells = [("freeze", 100), ("black", 25)]
        D = {}
        for c, L in cells:
            W(f"### {c} L={L}\n")
            W("| 規則 | 成功 | 每集推論次數 | Δ 對 TE h=1 [paired boot] | x/y（規則單贏／TE 單贏） | p（未校正） |"); W("|---|---|---|---|---|---|")
            sub = te[(te["corr"] == c) & (te.L == L)]
            ref = vec(sub[sub.rname == "Median TE h=1"])
            for rn in ("Median TE h=1", "Median TE h=10", "fixed n=50", "fixed n=100", "oracle n=50"):
                s = vec(sub[sub.rname == rn])
                if len(s) < 100 or len(ref) < 100: continue
                paired(s, ref); D[(c, L, rn)] = s
                if rn == "Median TE h=1":
                    W(f"| {rn} | {rate_ci(s.success.values)} | {s.replans_total.mean():.1f} | | | |"); continue
                x, y, p = mcnemar(s.success.values, ref.success.values)
                W(f"| {rn} | {rate_ci(s.success.values)} | {s.replans_total.mean():.1f} | {fd(s.success.values, ref.success.values)} | {x}/{y} | {p:.3g} |")
            W("")
        need = [("freeze", 100, "oracle n=50"), ("black", 25, "oracle n=50"), ("freeze", 100, "fixed n=50")]
        if all(((c, L, "Median TE h=1") in D and (c, L, rn) in D) for c, L, rn in need):
            res = []
            for c, L, rn in need:
                x, y, p = mcnemar(D[(c, L, "Median TE h=1")].success.values, D[(c, L, rn)].success.values)  # TE-only wins = x
                res.append((c, L, rn, x, y, p, D[(c, L, "Median TE h=1")].success.mean() - D[(c, L, rn)].success.mean()))
            adj = holm([r[5] for r in res])
            W("**F3（Holm，3 個，皆 Median TE h=1 對下列規則，雙尾）**\n")
            W("| 格 | 對象 | TE 單贏／對象單贏 | Δ(TE−對象) | p | p_Holm |"); W("|---|---|---|---|---|---|")
            for r, a in zip(res, adj):
                W(f"| {r[0]} L={r[1]} | {r[2]} | {r[3]}/{r[4]} | {r[6]:+.2f} | {r[5]:.3g} | {a:.3g} |")
            W("")
            t1 = res[0][6] > 0 and adj[0] < 0.05
            t2 = abs(res[1][6]) <= 0.10 and res[1][5] > 0.05
            t3 = res[2][6] > 0 and adj[2] < 0.05
            V(f"P-TE1（freeze L=100：TE h=1 > oracle n=50，Holm 後 p<0.05）：{'成立' if t1 else '不成立'}（Δ={res[0][6]:+.2f}，p_Holm={adj[0]:.3g}）")
            V(f"P-TE2（black L=25：TE h=1 與 oracle n=50 接近，|Δ|≤0.10 且 p>0.05）：{'成立' if t2 else '不成立'}（Δ={res[1][6]:+.2f}，p={res[1][5]:.3g}）")
            V(f"P-TE3（freeze L=100：TE h=1 > fixed n=50，Holm 後 p<0.05）：{'成立' if t3 else '不成立'}（Δ={res[2][6]:+.2f}，p_Holm={adj[2]:.3g}）")
            o_f = mcnemar(D[("freeze", 100, "oracle n=50")].success.values, D[("freeze", 100, "fixed n=50")].success.values)
            W(f"探索性：freeze L=100 oracle n=50 對 fixed n=50：{o_f[0]}/{o_f[1]}，p={o_f[2]:.3g}（oracle 單贏／fixed 單贏）。\n")

    W("## 5. 預先登記預測逐條核對\n")
    for v in verdict:
        W(f"- {v}")
    W("")
    open("results/p2b_table.md", "w").write("\n".join(out) + "\n")
    # merged raw csv of the unseen / dtype pieces
    for nm, pat in (("p2b_unseen.csv", "p2b_unseen_*.csv"), ):
        fs = sorted(glob.glob(P + pat))
        if fs: pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv("results/" + nm, index=False)
    for nm, fn in (("p2b_dtype.csv", "p2b_dtype.csv"), ("p2b_tc_clean.csv", "p2b_tc_clean_fixed.csv"), ("p2b_pilot.csv", "p2b_pilot.csv"), ("p2b_generic_clean.csv", "p2b_generic_clean.csv")):
        if os.path.exists(P + fn): pd.read_csv(P + fn).to_csv("results/" + nm, index=False)
    print("wrote results/p2b_table.md,", f"GPU {gpu_h:.2f} h")


if __name__ == "__main__":
    {"select": select, "table": table}[sys.argv[1]]()
