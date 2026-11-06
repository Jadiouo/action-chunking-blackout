# P2／P3 實作筆記：參數出處、與原文的差異、強度怎麼挑
日期：2026-11-04　程式：`code/corrupt.py`、`code/run_p2p3.py`、`code/p2p3.sh`、`code/calib_generic.py`、`code/calib_care.py`、`code/p2p3_stats.py`、`code/make_realistic.py`、`code/make_baselines.py`。結果表：`results/realistic_table.md`、`results/baselines_table.md`；資料：`results/realistic.csv`、`results/baselines.csv`；中間檔：`results/parts/p2p3_*`。
標記：「（文獻）」＝取自原文，出處見 `docs/baselines_spec.md`；「（自訂）」＝我們的決定，無原文依據；「（沿用）」＝沿用本專案前期設定。

## 0. 與指示不同、或需要先知道的事
1. **沒有改 `code/run_t3t4.py`。** 新增獨立的執行迴圈 `code/run_p2p3.py`（同樣自寫、同樣取 `predict_action_chunk` 的 100 步 chunk），舊檔與舊結果原封不動。回歸檢查改在新迴圈做：fixed n=25 L=0 seeds 0–4 成敗與 `results/full_grid.csv` 5/5 相同（`.scratch/p2/reg_fixed_L0.csv`）；Mean TE L=0 seeds 0–4 與 full_grid 的 `1+TE` 5/5 相同；oracle n=25 L=50 black、detect n=25 L=50 freeze 各 10 個 seed 與 `oracle.csv`／`detect.csv` 10/10 相同。
2. **同迴圈列與 `run_grid.py` 列在失明格會逐 episode 翻轉。** 新迴圈（以及 T3/T4b 的 `run_t3t4.py`）把動作轉 float64 再餵環境，`run_grid.py` 是 float32。無失明（成功率高）的格兩者一致（5/5），但中等成功率的失明格逐 episode 成敗翻轉約 6–14%（black L=50 n=25/50：5/100、2/100；freeze L=50 n=25/50：14/100、2/100，見 `results/parts/p2p3_regress_*.csv`），邊際成功率仍相近（19 vs 22、37 vs 37、53 vs 57、77 vs 79）。對照：`run_grid.py` 自己重跑 freeze n=25 L=50 只翻轉 2/100（`p2p3_regress_rungrid_freeze.csv`），所以這是兩個數值路徑的差，不是一般重跑雜訊。對策：所有 P2、P3 的規則對 fixed 的配對檢定一律用**同迴圈**的 fixed 列（`p2p3_p3_fixed_*`、`p2p3_regress_*`、`p2p3_real_*`）；`oracle.csv`／`detect.csv` 與新迴圈逐 episode 相同，可直接混用；`full_grid.csv` 只用於 L=0 的乾淨參照（無失明成功率高、兩路徑一致）與標為 `*_grid_*` 的對照列（不做檢定）。這個差異也意味著：docs/rulers.md 的「重跑雜訊約 1%」只適用於同一程式碼重跑，跨程式碼路徑比較失明格時應預期更大的翻轉。
3. **P2 主實驗每格 90 seeds（seed 0–89），不是 100。** 預算：指示「總量 ≤6000 episodes」；照字面 4 損壞 × 24 格 × 100 = 9,600 超標。縮法（跑之前決定，見 `results/parts/p2p3_selected.json`）：noise／occlude／delay 各做 fixed、oracle、generic-detect × n∈{25,50,100} × L∈{50,100}（18 格）＋ simple-detect 只做 n=50 × L∈{50,100}（2 格；它只在「像素均值 <1 或與前一幀逐位元相同」時觸發，對這三種損壞按設計不會觸發，這 2 格用來驗證它確實 0 次觸發），共 20 格 × 90 = 1,800 episodes／損壞；flicker 只做 n=50 × L∈{50,100} × {fixed, oracle, generic}（6 格，540）。合計 5,940 ≤ 6000。試跑（480）、偵測器診斷（180）、校準另計。
4. **只做 3 + 1 個損壞的完整網格，沒有任何損壞「無害」到可以略過**，但所有 P2 強度都比黑畫面溫和（見 §3）。
5. **通用偵測器第一版失敗，已替換**（見 §2.3）；舊診斷結果留在 `.scratch/p2/diag_T2_version/`，不進表。
6. 主實驗的 CARE τ 在第一輪 open-loop 校準偏低，改用 closed-loop 兩輪校準（§2.2）。

## 1. P2：損壞定義與參數
共同：只作用在策略看到的影像，窗 [t0, t0+L) 內，窗外立即回到真影像；`agent_pos` 不動；t0、env seed 與 T2 相同；影像尺度 [0,1]，top 相機 480×640；亂數由 (20000+env_seed, t) 決定（同 seed 同步數在所有規則下得到同一組雜訊）。

| 損壞 | 定義 | 參數 | 出處 |
|---|---|---|---|
| black／freeze | 全零影像／窗前最後一幀 | — | （沿用）T2，freeze 的定義＝Suh 2026（見 literature.md） |
| gaussian noise | x' = clip(x + N(0,σ²), 0, 1)，每步獨立抽樣 | 試跑 σ∈{0.1, 0.3, 0.6}；入選 **σ=0.6** | （自訂）。LIBERO-VPro §III-A3 只說 "visual noise" 無 σ；CARE、Median TE 無高斯雜訊 |
| 部分遮擋 | 純黑矩形，面積比例 a，長寬比同畫面（邊長 √a 倍）；位置 center 或 rand（中心均勻落在畫面中央 50%×50% 區域，依 env seed 抽一次） | 試跑 a∈{0.25, 0.5, 0.75} × {center, rand}；入選 **a=0.5、center** | （自訂）。參考：Median TE §IV-B 遮擋掃過 48/84 像素邊長；LIBERO-VPro 遮擋比例 25/50/75/100% 是對目標物而非畫面 |
| 延遲 | 窗內第 t 步看到第 max(t−d,0) 步的真影像 | 試跑 d∈{5, 25, 50}；入選 **d=50**（1 秒） | （自訂）。LIBERO-VPro §III-B 有三個延遲水準但數值讀不到；d=25 約等於 n=25 的一個查詢週期 |
| 間歇閃斷 | 窗內每步以機率 p 把整幀換成「低亮度＋雜訊幀」clip(0.2x + N(0, 0.05²), 0, 1)，其餘步為真影像；**不是全黑** | 試跑 p∈{0.3, 0.6, 1.0}；入選 **p=0.6** | 形狀（自訂，依任務清單「低亮度／雜訊幀，不用全黑」）；p 為自訂，數量級參考 Median TE §V-D 的空白幀密度 q/7≈0.29–0.57（原文是連續 burst，不是獨立機率）。原 baselines_spec §5 建議的 CARE packet-loss 形狀（複製前一幀的矩形）幾乎不改變影像，故未採用 |

**閃斷以「每步」定義而不是「每個查詢時刻」**：這樣 fixed n 只在查詢那一步碰到機率 p，oracle／偵測規則／TE 每步看到的影像都來自同一個隨機序列，可以配對。

### 1.1 強度怎麼挑（試跑，`results/realistic_table.md` §1）
- 設定：fixed n=50、L=50、30 個 seeds（120–149，與評估 seeds 0–99、偵測器校準 seeds 100–119 都不重疊）；乾淨參照＝同 seeds 的 L=0（24/30）。
- 規則（跑之前宣告）：每種損壞取 R = fixed 成功率／乾淨成功率最接近 0.5 且落在 [0.15, 0.85] 的強度；平手取較弱者。後來補一條（看過試跑後、跑主實驗前）：flicker 的 p=1.0 等於整段連續損壞，不算「間歇」，不參選。
- 結果（R）：noise σ=0.1/0.3/0.6 → 0.87/0.75/0.50；遮擋 center 0.25/0.5/0.75 → 0.75/0.62/0.62，rand → 0.83/0.71/0.67；延遲 d=5/25/50 → 0.96/0.92/0.62；閃斷 p=0.3/0.6/1.0 → 0.87/0.79/0.75。
- 入選：σ=0.6、遮擋 0.5 center、d=50、p=0.6（R=0.50/0.62/0.62/0.79）。每組只有 30 個 episode，R 的誤差約 ±0.15–0.2，只用來挑強度，不當結論。
- 沒有任何損壞「對 ACT 無害」，但所有強度的 R 都高於黑畫面（同格 R≈0.4）。

## 2. 偵測器
### 2.1 simple-detect（沿用）
T4b：黑畫面（所有相機像素均值 <1.0，uint8 尺度）或與上一步策略收到的影像逐位元相同。（沿用 `run_t3t4.detect_blind`。）

### 2.2 generic-detect（自訂，不針對任何一種損壞）
- 特徵：ACT 自己的 ResNet18 骨幹（`policy.model.backbone`）layer4 特徵圖，adaptive avg pool 成 3×4，展平為 6144 維。
- 分數：特徵對「乾淨幀子空間」的 PCA 重建誤差（K=16 個主成分）；PCA 只用乾淨校準 seeds 100–109（3,434 幀）。
- 閾值：乾淨 seeds 110–119（3,335 幀）分數的 99 百分位＝28.29（目標每步誤報率 1%；`results/parts/generic_detector.json`）。在評估資料上沒有調整任何東西。
- 規則：偵測到 → 不重規劃（同 oracle／simple-detect）；沒偵測到 → 照固定 n。每步獨立判斷，沒有時間平滑。
- 標籤：偵測器指標以「對策略有害」為標籤（§2.4），不是「損壞存在」。
### 2.3 為什麼是重建誤差：第一版失敗
第一版用子空間內的 Mahalanobis 距離。診斷（fixed n=50、L=50、seeds 120–149）發現它完全失效：窗內分數中位數（2.6–3.9）低於乾淨分數中位數（3.8），black 的偵測率 0、AUC≈0.5（黑畫面與雜訊幀的特徵落在乾淨主成分子空間之內，離「子空間之外」才遠）。改成 PCA 重建誤差。**這個替換是看過「合成探測幀」後做的**（6 個 seed 120–125 的靜止起始幀，加上 σ=0.6／0.3 雜訊、遮擋 0.5／0.25、全黑、閃斷 p=1 後比較 T²、SPE、kNN 距離等 7 個變體；用的是試跑用的 seeds，不是評估 seeds 0–99），沒有用評估資料調閾值；診斷資料與表中數字是替換後重跑的。
### 2.4 「對策略有害」標籤與偵測器指標（`results/realistic_table.md` §3b）
每個時間步以損壞影像與同一步真影像各預測一個 chunk，取前 50 個動作的平均 L2 距離 D（動作單位，弧度）；門檻 D*＝「乾淨影像加 ±1 灰階隨機抖動」造成的 D 的 99 百分位（取自診斷 episode 的窗外步），D>D* 為有害。診斷 episode：fixed n=50、L=50、seeds 120–149（每種損壞 30 個）。報偵測率（有害步中報警比例）、窗內無害誤報、窗外（乾淨）誤報、generic 的 AUC。另外 §3a 報主實驗內每個偵測規則的窗內偵測率、窗外誤報、延遲。

## 3. P3：文獻方法
### 3.1 Median TE（Jiang 2026，arXiv:2609.27167）
| 項目 | 本實作 | 出處 |
|---|---|---|
| 聚合 | 對所有 t_i ≤ τ ≤ t_i+99 的候選 a^(i)(τ)=A_i[τ−t_i] 逐維取中位數（numpy，偶數取中間兩個平均） | （文獻）§V |
| 查詢間隔 h | **h=1**（M=100，主）、h=10（M=10，次） | h 的選擇（自訂）；原文 h∈{1,2,3,4}（Diffusion Policy，T_p=16） |
| T_o | 1（ACT 單幀） | （自訂，原文 T_o=2） |
| 對照 Mean TE | w_i=exp(−0.01·i)、i=0 最舊（lerobot／ACT Algorithm 2）；與 `ACTTemporalEnsembler` 數值差 2e-8（隨機 chunk 單元測試）；跑 black/freeze（同迴圈）與 noise | （文獻）ACT；原文 Median TE §V 的 age 權重方向與 lerobot 相反，我們照 lerobot |
| 擾動 | 連續 L 步失明（black／freeze）或 P2 的 σ=0.6 雜訊，而非「每 7 個查詢中連續 q 個」 | （自訂）；見 baselines_spec §1.4 的差異 |
| 預先宣告的預測 | h=1：L=25 接近 oracle、L=50 平手、L=100 失效（baselines_spec §1.4 第 6 點，跑之前寫下） | （推論） |

### 3.2 CARE 式觸發（Yamabe & Sakuma 2026，arXiv:2609.32550）
| 項目 | 本實作 | 出處 |
|---|---|---|
| 規則 | Algorithm 1：新 chunk 對舊 chunk 未執行部分，逐步 r_i=‖a−â‖/(‖a‖+‖â‖+ε)，累積 Σγ^i r_i ≤ τ 的最長前綴為執行長度 H（至少 1）；舊 chunk 的對應動作＝A_prev[H_prev+i]（ACT chunk 第 0 項對齊當前步） | （文獻）式 (3)–(5)、Alg.1 |
| γ | 0.85 | （文獻）§6.1 |
| H_max | ⌊100/2⌋=50 | （文獻）§5.1 公式，K=100 |
| H_init | 1 | （文獻）附錄 C.4 |
| ε | 1e-8 | （自訂）原文未給 |
| 動作空間 | 14 維絕對關節位置，含兩個夾爪 | （自訂）原文未讀到 |
| τ | **重新校準**：主 0.30、敏感度 0.18、0.50 | （自訂）見下 |
| 擾動 | 連續 L 步失明，非單步 | （自訂）原文自承持續擾動效果有限（D.4） |

**τ 校準**（乾淨 seeds 100–119；`results/parts/care_tau_calib.json`、`p2p3_carecheck.csv`）：目標乾淨平均執行長度 0.6·H_max=30（主）、0.4·H_max=20、0.8·H_max=40（敏感度，對應原文 D.3 掃 τ）；理由：原文 Table 3 乾淨平均長度為 H_max 的 44–76%。第一輪用 open-loop 曲線（fixed n=30 排程）估 τ=0.107/0.175/0.354，但 closed-loop 實測平均 H 只有 13.4/19.4/32.8（因為縮短 H 會改變之後的決策點）；第二輪直接掃 closed-loop：τ=0.18/0.30/0.50/0.80 → H=20.9/31.5/38.5/42.3。最後採 τ=0.18（H≈21）、0.30（H≈31.5，主）、0.50（H≈38.5）；0.8 的平均 H 只到 42，達不到 40 以上，沒有用。乾淨成功率（20 個校準 seeds）0.50/0.70/0.75，明顯低於 fixed n=50（0.91）：CARE 在 ACT 上於乾淨情況就有代價，因為縮短 H 會碰到 T1 診斷的 chunk 開頭停頓。原文建議的 τ=1.0／1.5 在 ACT 上完全不適用（r_i 量級只有 1e-2，校準資料 R_0 中位數 0.019）。

## 4. 統計做法
- Wilson 95% CI；配對檢定 McNemar 精確（二項）＋Holm 校正。家族：P2 表＝主表所有「規則對 fixed」的比較（3 規則 × 各格 × 各損壞）；P3 表＝Median TE 與 CARE（主 τ）各對 {fixed n=50、fixed n=100、Mean TE、oracle n=50、simple-detect n=50（black/freeze）或 generic n=50（noise）} 的比較。
- 保留率 R＝失明成功率／同 n 乾淨成功率（n 規則用同 n 的乾淨 fixed；TE／CARE 用自己的 L=0）；恢復率 ρ＝(規則−fixed)/(乾淨−fixed)，分母 <0.10 記 n/r 不報；兩者的 CI 為 seed 的配對 bootstrap（4000 次，2.5/97.5 百分位）。
- 差距 <0.05 標 noise，不解讀（rulers.md）。但 §0 第 2 點指出跨程式碼路徑時翻轉更大，所以 P3 全用同迴圈列。

## 5. 結果摘要（數字見 `results/realistic_table.md`、`results/baselines_table.md`；此處只列結論，皆為同迴圈配對）
GPU：P2＋P3 的 episode 秒數加總 ≈ 8.0 小時（`results/parts/p2p3_*.csv` 7.0 h＋`.scratch/p2` 的煙霧／診斷／被替換的第一版診斷 0.95 h；實際牆鐘加總 8.05 h，含載入模型），在 12 小時預算內。P2 主實驗 5,940 episodes（試跑 480、診斷 180＋被棄用的一版 180、校準與 CARE 校準另計）。

### 5.1 P2：核心發現（失明期間不重規劃就能恢復）在合成損壞下大致重現
- fixed n=50、L=50 的保留率 R：noise σ=0.6 為 0.46、遮擋 0.5 為 0.49、delay 50 為 0.65、flicker 0.6 為 0.75（黑畫面同格 0.41）。
- 被告知損壞區間的 oracle（失明期間不重規劃）：L=50 時 n=25／50 的恢復率 ρ 為 noise 1.02／0.86、遮擋 1.05／0.85、delay 1.06／0.79，flicker n=50 為 0.70，成功率回到 75/90 附近（同乾淨水準）；L=100 只恢復一部分（ρ 約 0.3–0.55，與 black L=100 相同：失明長到 chunk 用完仍要被迫重規劃）。n=100 時 oracle 等於 fixed（chunk 剛好用完），不增益。
- 通用偵測器（不被告知）在 noise、遮擋、flicker 與 oracle 打平甚至更好：noise／遮擋 L=50 n=50 都是 75/90（ρ 0.86／0.85），flicker L=100 n=50 為 79/90 對 oracle 65/90（ρ 0.94 對 0.52），因為它只在壞幀上抑制重規劃，好幀上仍可重規劃。簡單偵測器（黑／凍結）對這些損壞 0 次觸發，成功率與 fixed 完全相同（設計使然，負對照）。
- **負結果：延遲 d=50 無法由影像偵測。** 延遲幀是真實的舊幀，單幀偵測器分數與乾淨幀不可分（診斷偵測率 3.2%，窗內偵測 episode 比例 0.07–0.13，約等於誤報），generic 規則 ρ≈0（−0.10 到 0.00），與 fixed 同；oracle 仍能恢復（L=50 n=25／50：75/90 對 40／53）。凍結（freeze）也是同樣的原因：generic AUC 0.54。
- 偵測器指標（有害標籤，診斷 episode）：generic 對 black／noise／遮擋偵測率 1.00，flicker 0.61（有害步比例 0.60，AUC 0.991），delay 0.03、freeze 0.03；窗內無害步誤報 ≤0.013；窗外步誤報 0.019–0.044（目標是 1%，評估時略高：窗外含失明後狀態已偏離乾淨分布的幀）。
- 注意：L=100 的 delay n=50 的 fixed 本身 R=0.67（分母 <0.10 的 ρ 不報／不可解讀），delay 在 n=100 時 fixed／oracle／generic 完全相同。

### 5.2 P3：Median TE 與 CARE
- 預先宣告（baselines_spec §1.4.6）：Median TE h=1 在 black L=25 接近 oracle、L=50 平手（明顯下降）、L=100 失效。實測 black：L=25 為 85/100（oracle n=50 88、fixed n=50 66；對 oracle McNemar 10/13，Holm 後無差；對 Mean TE 53/100 顯著較好 p_Holm=5e-6）；L=50 為 54/100（fixed n=50 37、oracle n=50 85）——比 fixed 高但 Holm 後不顯著（35/18），顯著低於 oracle（5/36）；L=100 為 5/100（失效，與 Mean TE 7、fixed n=50 8 無差）。與預測相符，不是被推翻。h=10 與 h=1 近似（86／53／8）。
- freeze 與預測有出入：Median TE h=1 在 freeze L=100 為 74/100，高於 fixed（54–56）與 oracle n=50（49），對 oracle p_Holm=0.007，對 fixed 未達 Holm 顯著（0.09／0.49）。原因（推論）：凍結幀是舊但「一致」的幀，大量候選共識仍落在合理範圍；而 oracle 在 L=100 時 chunk 用完被迫用凍結幀重規劃。h=10 在同格只有 63/100，所以這個優勢依賴每步查詢。
- noise σ=0.6：Median TE h=1 L=50 為 48/90（fixed n=50 37、Mean TE 38、oracle／generic 75），不顯著優於 fixed；L=100 為 14/90（失效）。
- CARE 式觸發（τ 重校準）：沒有幫助，且在失明格比 fixed 差。black L=50：τ=0.18／0.30／0.50 → 22／23／32 /100，fixed n=50 為 37，τ=0.30 對 fixed n=50 的 ρ=−0.26 [−0.49, −0.08]；freeze L=50：49／67／70 對 fixed n=50 77。乾淨成功率也低於 fixed n=50（0.74／0.84／0.89 對 0.91）。與 CARE 原文自承的限制一致：持續失明時新 chunk 一直拿到壞影像，一致性檢查把執行長度縮短，反而更頻繁用壞影像重規劃（baselines_spec §2.4 的預先宣告預測，成立）。
- 與 oracle／detect 比：被告知失明的 oracle 與影像偵測式 detect 在 black／freeze L≤50 恢復到 0.85–0.88（ρ 0.88–1.07，同 n）；Median TE 只在 L=25 追得上；CARE 完全追不上。

### 5.3 限制
- 強度是自訂、試跑僅 30 seeds；所有 P2 強度都比黑畫面溫和（R 0.46–0.75），oracle 的絕對增益因此比黑畫面小。
- 主實驗每格 90 seeds，相鄰格共用同一批 seed；Holm 家族很大（P2：規則對 fixed 的 3×格×損壞個比較），p 值偏保守。
- 通用偵測器的特徵取樣、K=16、閾值 99 百分位是我們的選擇，沒有掃參數；評估時窗外誤報 ≈2–4%。
- 跨程式碼路徑的逐 episode 翻轉（§0.2）會讓與 `full_grid.csv` 舊列的比較偏噪，已避免；但 L=0 乾淨參照仍取自 full_grid（成功率高、兩路徑一致）。
