# P2b 預先登記（白方補做）：預測、挑選規則、主對比
日期：2026-11-04。來源：`.scratch/white_p2.md` §4 三件事。**本檔在送任何 P2b GPU 工作之前 commit；之後不改預測與規則，只能在檔尾「偏離紀錄」補記。**
程式：`code/run_p2b.py`（包 `code/run_p2p3.py` 的 `run_episode`，迴圈原樣）、`code/corrupt_p2b.py`、`code/p2b.sh`、`code/make_p2b.py`。結果：`results/p2b_*.csv`、`results/p2b_table.md`。舊 CSV 不動。

## 0. 共同規則
- 迴圈：`run_p2p3.run_episode`（自寫 chunk 索引，動作 float64）。Insertion 用同一個迴圈，換 `--task insertion`（env、checkpoint、t0 分布與 `tasks.py` 一致）。
- Seeds：評估 0–89（Transfer Cube，TC）或 0–199（Insertion）；試跑 120–149；確認性重跑 150–249；100–119 是舊的偵測器／CARE 校準，不再用。
- 偵測器凍結：`results/parts/generic_detector.npz`，sha256 = `5193a96b546ffd97d4632bf2c1eb8b01b40b049748c77e62241cf600e244c2f2`（K=16，thr=28.29）。P2b 期間不重擬合、不改閾值、不改特徵；`run_p2b.py` 啟動時檢查雜湊，不符就中止。
- 統計：Wilson 95% CI；配對檢定用 McNemar 精確（雙尾二項）；Holm 只在下面各「家族」內做，其餘全部標「探索性」，只報未校正 p。R＝損壞成功率／同 n 同迴圈乾淨成功率；ρ＝(規則−fixed)/(乾淨−fixed)；兩者 CI 用 seed 配對 bootstrap（4000 次，2.5/97.5 百分位）；分母 <0.10 不報 ρ（記 n/r）。差距 <0.05 不解讀為差異（rulers.md）。
- 預算：總 GPU ≤3.5 h（以 episode `seconds` 加總）。估計 #1 Insertion 約 1.7 h、#1 TC 約 0.2 h、#2 約 0.5 h、#3 約 0.5 h，合計約 2.9 h。若實際累計超出，先砍 #1 Insertion 的 L=100 格，再砍 Insertion 乾淨參照 n=100。

## 1. 事項 #1：同迴圈對照與 dtype 消融
### 1.1 Insertion 同迴圈 fixed
- 內容：`run_p2b.py --task insertion --rule fixed`，n∈{25,50,100}×L∈{50,100}×{black,freeze}×200 seeds（2400 episodes）；乾淨參照 L=0、n∈{25,50,100}×200 seeds（600）。oracle／detect 沿用 `results/insertion.csv`（它們本來就是 `run_t3t4.py` 的 float64 迴圈，與 `run_p2p3` 同路徑；P2 已驗證 10/10 逐 episode 相同）。
- 預測：(a) 邊際成功率可重現：同迴圈 fixed 與舊 `run_grid.py` fixed 在每格相差 ≤0.05；(b) 逐 episode 翻轉出現在中間成功率格（舊 fixed 成功率 0.1–0.9 的格，Insertion 的乾淨成功率只有約 0.2，所以翻轉絕對數量小）；(c) 結論不變：黑畫面 L=50，oracle 與 detect 對同迴圈 fixed 仍然顯著提升（方向與舊相同）；凍結 L=50 仍無差異（|Δ|<0.05）；L=100 仍不顯著或微弱。
- **主對比家族 F1（Holm，3 個）**，指標 reward=4（預先宣告的主指標），皆 black、L=50、200 seeds，對象為同迴圈 fixed：
  1. oracle n=25 對 fixed n=25（舊邊界格，Holm 家族內 p=0.0415）
  2. detect n=25 對 fixed n=25（舊 p=0.0233）
  3. oracle n=50 對 fixed n=50
  「結論不變」的判準＝三者方向與舊相同，且 Holm 後 p<0.05。若第 1、2 項掉到 Holm 後 ≥0.05，論文把 n=25 黑畫面格降級為「邊界、未確認」。其餘格（freeze、L=100、reward≥3 事後指標）全部探索性。
### 1.2 TC 乾淨參照與 dtype 消融
- TC 同迴圈 fixed，L=0，n∈{25,50,100}×90 seeds（0–89）：預測與 `full_grid.csv` 的乾淨成功率每 n 相差 ≤0.05。
- dtype 消融：新迴圈把動作轉成 float32 再餵環境（`--act_dtype float32`），TC n=25 L=50，freeze 與 black 各 100 seeds（seeds 0–99）。對照：舊 `full_grid.csv`（`run_grid.py`，float32）與 float64 的 `results/parts/p2p3_regress_{black,freeze}.csv`（n=25 列，翻轉 5/100、14/100，合計 19/200）。
- 判準（逐 episode 成敗與 `run_grid.py` 列不同的數量，兩格合計，分母 200）：
  - ≤8/200 → 「dtype 是主因」已驗證（`run_grid.py` 自己重跑 freeze 2/100，作為雜訊底線）；
  - ≥14/200 → 「dtype 不是主因（或不是唯一原因）」，論文寫成未解；
  - 9–13/200 → 部分，寫「部分由 dtype 解釋」。
- 我的預測：翻轉大幅下降（≤8/200）。若不成立照實報。

## 2. 事項 #2：未見過損壞與乾淨代價
### 2.1 兩種新損壞（設計偵測器時從未看過，也沒進過 7 個變體的探測集）
- blur σ：高斯模糊（`cv2.GaussianBlur`，核大小自動，σ 為像素數，畫面 480×640）。試跑強度 σ∈{5, 15, 40}。
- dim g：亮度／對比下降，x' = round(g·x)，無雜訊。試跑強度 g∈{0.5, 0.25, 0.1}。
- 窗口與 TC 其他損壞相同：[t0, t0+L)，窗外立即回真影像；t0、env seed 與 T2 相同。
### 2.2 強度挑選規則（現在寫死，與 P2 的規則相同，不得事後增減）
- 試跑：fixed n=50、L=50、seeds 120–149（30 個）；乾淨參照＝同 seeds、同迴圈的 L=0 fixed n=50。R＝損壞成功率／乾淨成功率。
- 規則：取 R 最接近 0.5 且落在 [0.15, 0.85] 內的強度；平手取較弱者。
- 若三個強度 R 都 >0.85：再跑一個預先宣告的更強強度（blur σ=80；dim g=0.03），在四個中依同一規則挑；仍全部 >0.85 → 取最強者並標記「對 ACT 近乎無害」，ρ 多半 n/r，照報。若三個 R 都 <0.15 → 取最弱者。
- 沒有其他補丁。
### 2.3 主實驗
- 每種損壞：fixed、oracle、generic-detect × n=50 × L=50 × 90 seeds（0–89）。共 540 episodes。
- 乾淨代價：generic-detect L=0 × n∈{25,50,100} × 90 seeds，對照 §1.2 同迴圈 fixed L=0（同 seeds 配對）。
- 偵測器指標：窗內被標記的步數比例（det_in/steps_in）、窗內至少偵測一次的 episode 比例、窗外誤報率（det_out/steps_out）、至少一次窗外誤報的 episode 比例。
### 2.4 預測（偵測器已凍結，這些數字現在寫下）
- dim：窗內標記步數比例 ≥0.90；在 fixed 損傷足夠（ρ 分母 ≥0.10）時 ρ_generic ≥0.60。
- blur：窗內標記步數比例 ≥0.50（我對這個較沒把握，預期比 dim 低）；ρ_generic ≥0.40（分母 ≥0.10 時）。
- 兩者窗外誤報率 ≤0.05（舊值 0.019–0.044）。
- 乾淨代價：generic L=0 與同迴圈 fixed 的成功率差 |Δ|<0.05（n=25、50、100 各自）。
- 什麼算「偵測器不通用」：任一新損壞在 fixed 損傷足夠時 ρ_generic <0.30，或窗內標記比例 <0.30。這種結果會直接進論文，並把「通用」改稱「從乾淨幀學的 OOD 偵測器，只對四種舊損壞與 dim 驗證過」。
- **主對比家族 F2（Holm，2 個）**，指標 success：generic n=50 對 fixed n=50（L=50）在 blur、在 dim 各一。其餘（oracle 對 fixed、generic 對 oracle、乾淨代價的 McNemar）探索性。

## 3. 事項 #3：Median TE 對 oracle 的確認性重跑
- 內容：TC，seeds 150–249（100 個，從未用過），兩格 freeze L=100 與 black L=25；規則 Median TE h=1、h=10、fixed n=50、fixed n=100、oracle n=50（皆同迴圈，同 seed／t0 配對）；共 1000 episodes。
- 預測（舊值：freeze L=100 為 TE h=1 74/100、oracle n=50 49、fixed n=50 54；black L=25 為 TE h=1 85、oracle n=50 88、fixed n=50 66）：
  - P-TE1：freeze L=100，Median TE h=1 的成功率高於 oracle n=50。
  - P-TE2：black L=25，Median TE h=1 與 oracle n=50 接近（|Δ|≤0.10，且未校正 McNemar p>0.05）。
  - P-TE3：freeze L=100，Median TE h=1 高於 fixed n=50（方向；舊資料未校正 p=0.0017、Holm 後 0.0945，這次當作確認）。
- **主對比家族 F3（Holm，3 個）**，皆 Median TE h=1，雙尾：
  1. freeze L=100：TE h=1 對 oracle n=50
  2. black L=25：TE h=1 對 oracle n=50（預期不顯著）
  3. freeze L=100：TE h=1 對 fixed n=50
  P-TE1 成立＝項 1 方向為 TE>oracle 且 Holm 後 p<0.05；P-TE2 成立＝項 2 的 |Δ|≤0.10 且 Holm 前 p>0.05；P-TE3 成立＝項 3 方向為 TE>fixed 且 Holm 後 p<0.05。P-TE1 不成立 → 論文把 freeze L=100 的優勢降級成「單一格、未重複」。
- 探索性（只報未校正 p）：oracle n=50 對 fixed n=50（freeze L=100，預期 oracle 不優於 fixed）；TE h=10 對 h=1；fixed n=100。
- 計算量一併報告（每集推論次數：TE h=1 約 366、h=10 約 37、oracle 約 6.5）。

## 4. 偏離紀錄（跑的過程中補記，只增不改）
- 2026-11-06：GPU 實際 3.74h，超過預算 3.5h 約 0.24h，沒有砍 #543、#542。全部工作在主 session 發現超出前就已跑完。
- 2026-11-06：dim 入選 g=0.1（R_pilot=0.79），是全部強度中唯一落在 [0.15, 0.85] 的，也就是規則內傷害最小的選擇；所以 dim 的 ρ 分母較小。
- 2026-11-06：make_p2b.py 的 table 由主 session 執行，因為 P2b agent 已被暫停；抽驗 dim／blur generic 與 Insertion new fixed n=50 L=50 與表一致。
