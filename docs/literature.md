# Literature：ActionChunking（相機失明下的 Action Chunking）
日期：2026-11-03

來源（原始工作檔，不在本 repo）：`.scratch/lit/1.md`（引用核實）、`2.md`（執行步數／重規劃）、`3.md`（相機失效穩健性／runtime monitoring）、`4.md`（投稿場合）。已去重。所有 2026 年 arXiv 論文都是預印本，未必經同行審查。
證據等級：A＝讀過全文或相關段落；B＝只讀摘要（一手頁面）；C＝只有題錄或背景知識。「只讀摘要」「讀過全文」標記沿用來源檔。標「推論」者是來源檔作者的判斷，不是論文自己說的。

## 摘要

### 最接近的做法與差異
- **Blackout vs Freeze（Suh 2026，2609.39145，讀過全文）**：在 π0.5、GR00T 兩個 VLA 上比較全黑與凍結，LIBERO＋WidowX 真機。差異：(1) 模型與任務不同（3B 級 VLA，非 ACT；非 Transfer Cube）。(2) 重規劃間隔固定（π0.5 每 5 個動作、GR00T 每 8 個），沒有掃執行步數 n，也沒有重規劃觸發規則。(3) 故障持續到 episode 結束，恢復條件「untested」，沒有有限長度失明後恢復。(4) 本體感覺是策略輸入（有／無各自訓練），我們是重規劃觸發器。(5) 重點在物理失敗型態（關節極端度、掉物、接觸），我們量成功率與重規劃代價。它有近似的介入（oracle camera-token masking、本體感覺消融），報告措辭需承認。
- **MAIL-Bench／MINT（Jiang 2026，2609.35003，讀過全文）**：RoboCasa365 上在策略完成時間 30–60% 處中斷某相機直到結束；MINT 用缺視角訓練，推論時補視角（光流外推或世界模型），不可靠時撤回預測視角。差異：它的行為改變是補觀測或標記缺失，我們是失明期間不重規劃、繼續舊 chunk，免補影像、免重訓；它刻意不用空白或凍結影像（缺幀以缺席方式送入），所以沒有 black vs freeze；模型是 π0.5、GR00T N1.5，非 ACT。審稿人幾乎必然拿 MINT 比（推論）。
- **Median Temporal Ensembling（Jiang 2026，2609.27167，部分全文）**：把 temporal ensembling 的平均換成逐座標中位數，對被污染的預測免訓練穩健；空白相機幀實驗只在 Diffusion Policy、robomimic can、一個 stride（h=2）上做（原文 §V-D、§VIII）；ACT 只出現在對抗性攻擊的實驗，不是空白幀。差異：它是穩健聚合、不依賴偵測，擾動是稀疏查詢（q/M）而非連續失明；只評估一個 stride，沒有掃 execution horizon；單人預印本。可作我們「不重規劃」方案的對照 baseline。
- **Zeng 2026（2608.15938，部分全文）**：受控掃 execution horizon（模擬 Texec 1–15，真機 2/4/8），主張長 open-loop 的好處來自專家非馬可夫（停頓）而非 compounding error。差異：觀測品質永遠正常，沒有損壞或擾動實驗（已讀範圍內）；主張足夠長 context 下最 reactive 的 closed-loop 最好，我們的失明結果顯示壞觀測下重規劃有害，條件不同。可用來解釋 n=1 失敗。
- **CARE（Yamabe 2026，2609.32550，讀過摘要與方法段）**：單步嚴重觀測擾動，chunk 執行越長傷害越大；依新舊 chunk 重疊段一致性動態選執行長度，不一致就縮短。差異：它是一步瞬間擾動，我們是持續失明；方向相反（推論）：一步壞觀測應縮短執行盡快重規劃，持續失明時重規劃有害所以應不重規劃，兩者並不矛盾，失明持續時間決定最佳策略。

### 「已被做過」
- 系統性掃 execution horizon 對成功率的影響（Zeng 2026、PACE、AutoHorizon、ACT 與 Diffusion Policy 原文）。共同結果：非單調、倒 U 形、與任務有關。
- 依情況決定何時重規劃：非常擁擠（熵、注意力、denoising 變異、一致性、RL 學出來、世界模型、視覺偏差觸發；見下方論文條目與「其他相關」）。
- 比較全黑與凍結兩種相機故障（Suh 2026；Median TE 在 robomimic 上有並排的間接證據）。
- 相機中途失效下的閉環操作與 benchmark（MAIL-Bench、LIBERO-VPro），含 blackout 與 camera staleness 維度。
- 失效偵測與 runtime monitoring（Sentinel、FIPER、FAIL-Detect），只偵測、不規定偵測後怎麼做。
- 訓練時的 sensor／modality dropout（Bohez 2017、Skand 2024、Targeted Modality Dropout 2026）。
- 本體感覺作為補償視覺故障的輸入（Suh 2026）。

### 看起來是新的（在這次檢索範圍內查無；非保證不存在）
- 失明期間用壞影像重規劃的代價隨 n 變大（機制歸因為「用壞影像重規劃」；CARE 與 Median TE 只是旁證）。
- 失明期間不重規劃可救回（查無任何一篇明確提出並量化「失明時不重規劃、繼續舊 chunk」；最近的是 MINT 的撤回視角）。
- 本體感覺（指令與位置誤差）觸發重規劃無效（Suh 把本體感覺當輸入而非觸發器；自適應 horizon 家族的觸發訊號都來自策略內部或預測本身，沒有一篇用本體感覺，也沒有一篇把觀測品質變壞當評估情境）。
- 有限長度失明後恢復（Suh 的故障持續到底、恢復未測；MAIL-Bench 中斷直到結束）。
- 另：「預訓練 ACT 只執行每個 chunk 第 1 步（n=1）完全失敗，因 chunk 開頭停頓」查無明確報告；最近的是 ACT／Diffusion Policy／Zeng 對示範停頓的討論，以及 RTC 對 chunk 邊界停頓（推論延遲造成，機制不同）的描述。

### 要注意的衝突
- **Blackout vs Freeze 方向相反**：Suh 在 VLA 上 freeze 比 black 更糟（scene frozen 5.8% 對 scene black 28.9%，π0.5，n=450）；我們在 ACT 上 black 比 freeze 更糟（24 組配對中 13 組）。報告裡值得提並給假設（推論，未驗證）：它的 freeze 多半凍結在 episode 起點畫面，策略看到「永遠沒進展」的場景，我們的 freeze 凍結在失明開始時的合理畫面；黑幀對 ACT 是分布外輸入，而 VLA 有大規模預訓練視覺。Median TE（Diffusion Policy，robomimic can）與我們方向一致（空白幀比整段 burst 凍結幀傷得多，後者最多損失 0.08；ACT 沒有做這組空白對凍結的比較），但其擾動是稀疏查詢，不是連續失明，不宜直接類比。
- **ActFovea 可能與「本體感覺觸發無效」衝突**：ActFovea（2607.29169，只讀摘要）用 kinematics、本體感覺與近期動作檢查視覺與觀測新鮮度，並處理 frozen-observation replay。它的觸發條件與我們不同之處需讀全文才能判斷（未讀）。在讀全文前，不要在報告中宣稱「本體感覺觸發普遍無效」，限定為「我們的指令與位置誤差觸發在此設定下無效」。
- **來源檔之間的差異已結案（2026-11-04 對照原文 arXiv:2609.27167v1 HTML §IV-B、§V-D、§VIII）**：整篇範圍是 robomimic lift／can／square 與 PushT、四個 stride；空白幀實驗只在 Diffusion Policy、can、一個 stride（h=2）；乾淨成功率 Diffusion Policy 1.00、ACT（以 can 訓練）0.85，均為 h=2，兩者各指不同策略，不衝突；ACT 的 0.010→0.595 是 20 步對抗性攻擊（q=2），不是空白幀。

### 建議採用／避開
- 最強已知 baseline：(a) 持續重規劃（n 小、每步查詢）與 ACT temporal ensembling（模型卡 n=100 為 83%）；(b) Median TE（穩健聚合）；(c) MINT 式撤回或缺視角（需重訓或補視角，只能概念對照）；(d) CARE 式一致性縮短執行長度。
- 採用：LIBERO-VPro 的擾動分類（blackout、雜訊、常數影像、過期重播）與措辭（我們的 freeze 對應 Camera Staleness）；MAIL-Bench 的條件成功率配對協定；FAIL-Detect 的 conformal 校準閾值協定。
- 避開：不宣稱第一個做 black vs freeze；不宣稱第一個用本體感覺（Suh 已作為輸入）；不把 ACT 論文當 83% 的出處（見下方）。定位建議：「在 chunking 策略上，失效時最便宜的緩解是不重規劃，不需補影像或重訓」。

## 論文

### 1. Blackout vs Freeze（最近鄰）
- 引用：Heejae Suh, Jongwook Han, Zahra Gholami, Yohan Jo, 2026, "Blackout vs. Freeze: Analyzing Physical Failure Modes of VLAs under Camera Faults", arXiv:2609.39145（v1，2026-09-30，cs.RO）。https://arxiv.org/abs/2609.39145 ；全文 https://arxiv.org/html/2609.39145v1
- 一句貢獻：在 π0.5 與 GR00T 上比較影像全黑與凍結，兩者成功率都很低但物理失敗型態不同（凍結造成更極端關節動作；抓取後全黑造成更多掉落），並評估兩種緩解（20% 訓練樣本換黑幀；免訓練 mean-embedding replacement）。
- 與本專案關係：最近的做法，細節見摘要。不掃 execution horizon、故障持續到底。主要數字（n=450，有本體感覺）：π0.5 乾淨 93.1、scene black 28.9、wrist black 13.1、scene frozen 5.8、wrist frozen 3.6；GR00T 乾淨 98.7、scene black 51.6、wrist black 12.2、scene frozen 9.3、wrist frozen 0.0。每條件單 seed。
- 可借：故障定義（black＝零 RGB；freeze＝重複故障開始瞬間的畫面）；附錄 oracle camera-token masking（scene freeze 1.3%→47.3%）可對應我們的 oracle 條件；「低成功率可掩蓋差異很大的接觸／力」的論點。
- 證據等級：讀過全文（HTML，經摘要器處理，非逐字；數字引用前建議對照原 PDF 的 Table 5）。

### 2. MAIL-Bench／MINT
- 引用：Mingle Jiang, Rui Xu, Yunke Wang, Chang Xu, 2026, "Learning to Act under Visual Interruptions with Vision-Language-Action Models", arXiv:2609.35003（2026-09-28）。https://arxiv.org/abs/2609.35003 ；benchmark 頁 https://minglejiang.github.io/Mail-Bench/
- 一句貢獻：MAIL-Bench（RoboCasa365 18 任務、900 固定場景，在策略自己完成時間的 30–60% 處中斷某相機直到結束）與 MINT（缺視角訓練＋推論時補視角，不可靠時撤回預測視角）。
- 與本專案關係：問題設定最接近（中途相機失效、預訓練策略），行為改變路線不同（補或撤回 vs 不重規劃）；π0.5、GR00T N1.5＋AgiBot G2 真機；結果顯示失去單一腕部相機可能比失去兩個第三人稱相機更傷。
- 可借：條件成功率協定（以策略自己的成功 rollout 為參考、在其完成時間 30–60% 注入、只在完整相機成功的場景上算）；失效分角色（腕部／第三人稱／全部）；「撤回不可靠預測」概念。
- 證據等級：A（讀過全文）。

### 3. Median Temporal Ensembling
- 引用：Yuhang Jiang, 2026, "Median Temporal Ensembling: Training-Free Robust Aggregation for Action-Chunked Visuomotor Policies", arXiv:2609.27167（2026-09-22）。https://arxiv.org/abs/2609.27167 ；專案頁 https://avalon-s.github.io/MedianTE/
- 一句貢獻：對重疊 chunk 預測用逐座標中位數取代指數加權平均，q 個候選被汙染時有組合式保證；在相機影格到達時是空白的故障下（Diffusion Policy，robomimic can）可部分恢復表現。
- 與本專案關係：唯一碰到「空白影格＋temporal ensembling」的工作。空白影格（Diffusion Policy，robomimic can，乾淨 1.00，h=2）密度 q=2 時 mean 0.510、median 0.690（成對勝負 12:48，p=3.2e-6）；q=3 時 0.120 對 0.185；q=4 皆為 0。單一過期幀、間歇遮擋、動態模糊、整段 burst 凍結對無防禦的 Diffusion Policy 最多只損失 0.08。ACT（以 can 訓練，乾淨 0.85）只出現在 20 步對抗性攻擊實驗：q=2 時 0.010→0.595、q=3 時 0.000→0.175；這不是空白幀。Limitations（原文 §VIII）：空白影格只在一個任務、一個 stride 評估，沒有掃 execution horizon。
- 可借：同一協定下「空白幀比凍結幀傷得多」的獨立證據，可與我們的 black／freeze 對照；作為「不重規劃」方案的對照 baseline（temporal ensembling 在 n=1 時等價於每步重規劃並平均）；zero-fill on timeout 可引用來說明 black 是真實的相機失效型態。
- 證據等級：A（2026-11-04 重讀 arXiv HTML v1 的相關段落與 abs 頁）。單人預印本，證據較弱。

### 4. Zeng 2026：重訪 open-loop 執行
- 引用：Zeng, Agarwal, Bati, Lee, et al., 2026, "Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies", arXiv:2608.15938。https://arxiv.org/abs/2608.15938
- 一句貢獻：4 個模擬＋2 個真機任務受控掃 execution horizon，主張長 open-loop 的好處主要來自專家非馬可夫（含停頓）而非 compounding error。
- 與本專案關係：最接近「系統性量測 execution horizon」。馬可夫專家下 Texec*=1 最佳、成功率隨 Texec 單調降；人類示範加短 context（To=2）時呈倒 U 形，Texec=6 最佳；短 horizon 失敗模式含機器人徘徊、重複對齊而不承諾。沒有觀測損壞實驗（已讀範圍內）。
- 可借：「專家是否馬可夫」「context 長度」「Texec」交叉掃描協定；用來解釋我們 n=1 失敗（ACT 預設 To=1，屬短 context；若「chunk 開頭停頓」成立，是此現象在預訓練 ACT 上的具體機制，推論）。
- 證據等級：A（部分全文，讀 HTML 前 100k 字元，未讀剩餘 54k 字元）。

### 5. CARE
- 引用：Shojiro Yamabe, Jun Sakuma, 2026, "Are Vision-Language-Action Models Robust to One-Step Observation Perturbations?", arXiv:2609.32550（2026-09-26）。https://arxiv.org/abs/2609.32550
- 一句貢獻：單步嚴重觀測擾動就大幅降低 VLA 表現且影響隨 chunk 執行長度而變；CARE 依新預測與上一 chunk 未執行部分的一致性動態選執行長度，不一致時縮短。
- 與本專案關係：「chunk 長度 vs 壞觀測」的直接文獻；與我們方向相反但不矛盾（見摘要）。
- 可借：一致性訊號可當第三種偵測器，與「全黑／與前幀相同」簡單偵測並列比較；單步擾動協定、「在脆弱時刻注入」；可變成一個實驗：失明長度掃描，找最佳策略從縮短轉為保持的交叉點（推論）。
- 證據等級：A（摘要與方法段）。

### 6. ActFovea
- 引用：Yu et al., 2026, "ActFovea: Runtime Safeguarding for VLA Policies via Spatiotemporal Visual-Action Consistency", arXiv:2607.29169。https://arxiv.org/abs/2607.29169
- 一句貢獻：用 kinematics、本體感覺與近期動作建立 action-conditioned 視野，檢查視覺運動與觀測新鮮度是否與動作一致；對可恢復擾動嘗試恢復，對觀測停滯或重播觸發有界安全失敗程序（π0 在 LIBERO 局部視覺覆蓋下 49.3%→90.3%，論文自己說）。
- 與本專案關係：評估情境含 visual delay 與 frozen-observation replay，與我們的凍結相同；偵測用到本體感覺一致性，可能與我們「本體感覺觸發無效」衝突，需讀全文比較觸發條件。摘要未寫是否縮短 horizon（網搜摘要曾提到，未確認）。
- 可借：凍結畫面（觀測新鮮度）偵測思路；本體感覺＋視覺一致性。
- 證據等級：B（只讀摘要）。

### 7. LIBERO-VPro
- 引用：Huiqiong Li, Zhiting Mei, Anirudha Majumdar, Jingjing Chen, Yu-Gang Jiang, Bin Zhu, 2026, "LIBERO-VPro: Benchmarking Closed-Loop Visual Robustness of Robotic Foundation Models", arXiv:2609.24350（2026-09-21）。https://arxiv.org/abs/2609.24350
- 一句貢獻：四維度（Visual Evidence Degradation、Camera Staleness、Visual Source Consistency、Task-Relevant Scene Variation）、12 類挑戰、96 設定、3,296 任務條件、約 196,000 模擬 episode，加 Franka 真機 200 次 rollout。
- 與本專案關係：Temporary Observation Corruption 以 blackout 或雜訊注入整幀，可控制頻率與操作階段；Camera Staleness 用歷史重播模擬延遲、丟幀、非同步。結論：模型對缺失或過期觀測高度敏感，但對 stale 的穩健性隨任務與模型而異。評測對象是 VLA 與世界動作模型，不是 ACT；沒有偵測後改變行為的緩解；沒讀到 blackout 與 staleness 的直接並排比較。
- 可借：擾動分類與措辭（我們的 freeze 對應 Camera Staleness）；「發生階段」作為自變數。
- 證據等級：A（擾動定義與摘要；對比結果細節未讀完）。

### 8. ACT（我們的策略）
- 引用：Tony Z. Zhao et al., 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023, arXiv:2304.13705。https://arxiv.org/abs/2304.13705
- 一句貢獻：提出 action chunking（預測 k 步）與 temporal ensembling（每步查詢、對重疊 chunk 指數加權平均）。
- 與本專案關係：我們直接使用的策略與 baseline。論文自己說（讀 PDF 全文）：訓練不同 k、關閉 temporal ensembling，成功率平均從 k=1 的 1% 升到 k=100 的 44%，k=200、400 略降。支持「小 n 差、過大 n 也降」；但其 k=1 是訓練時 k=1，我們是預訓練 chunk 模型只執行 n=1，機制不同。Table I：Cube Transfer（sim）人類示範資料 Transfer 50%、腳本資料 86%（3 seeds × 50 評估），**不是 83%**。
- 可借：n 的取捨論述；「示範停頓」的時間相關混淆因子是我們「chunk 開頭停頓」假說的最近一手線索（ACT 沒有量測 chunk 內位置）。
- 證據等級：A（PDF 全文）。

### 9. Diffusion Policy
- 引用：Cheng Chi et al., 2023/2024, "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", IJRR（期刊版，卷期本次未驗證），arXiv:2303.04137。https://arxiv.org/abs/2303.04137
- 一句貢獻：receding-horizon 動作序列預測，並做 action horizon（Ta）取捨與延遲強健性實驗。
- 與本專案關係：action horizon 大於 1 有助一致動作、補償示範 idle 段落，太長因反應慢而降低表現，多數任務最佳 Ta=8；idle actions 讓單步策略（BC-RNN、IBC）在真機卡住；延遲最多 4 步仍維持峰值表現。與「n 太小會失敗」一致，但沒有量測「預訓練後只取第 1 步」，也沒有觀測失效。
- 可借：n=1 失敗的背景引用；延遲強健性評估協定。
- 證據等級：A（PDF 全文）。

### 10. PACE
- 引用：Nie et al., 2026, "PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking", arXiv:2606.00537。https://arxiv.org/abs/2606.00537
- 一句貢獻：成功率隨 execution horizon 與任務有關且非單調（論文自己說）；只用預測 chunk 的速度剖面，在低速轉換點作為候選重規劃邊界（training-free）。RoboTwin2.0 50 任務 57.8%→64.2%；真機 ALOHA/Franka 50.7%→70.4%。
- 與本專案關係：掃 horizon＋自適應重規劃；沒有觀測失效情境。
- 可借：以 chunk 速度剖面決定截斷點。推論：低速點恰好對應停頓位置，若我們的 n=1 失敗來自 chunk 開頭停頓，「在低速點重規劃」有相反風險，值得做對照分析。
- 證據等級：B（只讀摘要）。

### 11. AutoHorizon
- 引用：Wang et al., 2026, "VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies" (AutoHorizon), arXiv:2602.21445（Springer 章節 DOI 10.1007/978-3-032-37196-6_33）。https://arxiv.org/abs/2602.21445
- 一句貢獻：改變 execution horizon 造成顯著差異且先升後降；chunk 內動作對視覺語言 token 的 attention 幾乎不隨動作位置改變，首尾動作是穩定錨點；用 action self-attention 估每個 chunk 的 horizon。
- 與本專案關係：量測＋自適應；重疊於「n 先升後降」；觀測失效未涉及。
- 可借：attention 作為「模型自認的預測極限」訊號；可檢查 ACT 的 cross-attention 在黑畫面時是否有可偵測變化（推論，未驗證）。
- 證據等級：B（只讀摘要）。

### 12. VLA-Corrector
- 引用：Pan et al., 2026, "VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon", arXiv:2607.01804。https://arxiv.org/abs/2607.01804
- 一句貢獻：Latent-space Vision Monitor 比較預測與實際視覺特徵演變，偵測到持續偏離就截斷剩餘動作並修正重規劃，形成事件觸發的自適應 horizon。
- 與本專案關係：事件觸發重規劃，直接對應「何時重規劃」；觸發訊號是視覺動力學偏離，與我們測過的本體感覺觸發（無效）形成對照。失明時視覺特徵本身就壞，摘要未談其表現（推論：凍結畫面會讓「實際」特徵不變，可能被判偏離也可能不被判）。
- 可借：視覺偏離觸發器作為本體感覺觸發的對照。
- 證據等級：B（只讀摘要）。

### 13. Lazzati 2026：為何 action chunking 有效
- 引用：Lazzati, Stachowicz, Chen, Metelli, Wagenmaker, Levine, 2026, "Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?", arXiv:2608.02547。https://arxiv.org/abs/2608.02547
- 一句貢獻：常見解釋（時間一致性、較短有效 horizon、表徵）不足；chunking 靠較強的非馬可夫表達力與較少 compounding error，另有 implicit ensembling 效應；隨機延遲 k 步的單步策略集成可達到大部分好處（論文自己說）。
- 與本專案關係：支持「n=1 單步＝失去 chunk 帶來的非馬可夫能力」，與 Zeng 互補。
- 可借：延遲策略集成作為 n=1 的替代 baseline。
- 證據等級：B（只讀摘要）。

### 14. RTC（Real-Time Chunking）
- 引用：Kevin Black, Manuel Y. Galliker, Sergey Levine, 2025, "Real-Time Execution of Action Chunking Flow Policies", NeurIPS 2025, arXiv:2506.07339。https://arxiv.org/abs/2506.07339
- 一句貢獻：目前 chunk 執行時非同步生成下一個 chunk，已承諾動作固定、其餘用 inpainting，免重訓，對推論延遲穩健（Kinetix 12 動態任務＋6 個真機雙臂任務）。
- 與本專案關係：處理觀測延遲而非損壞；「已承諾的動作前綴不改，新資訊只影響後面」與「失明期間不重規劃」機制相鄰。摘要指出推論延遲造成 chunk 邊界停頓，是文獻中唯一明說「chunk 邊界停頓」的一手來源，但原因是推論延遲，不是示範停頓，我們模擬為同步推論，引用要區分。RTC 假設新觀測是好的。
- 可借：延遲／過期類文獻代表；延遲幅度當自變數的 benchmark 做法。
- 證據等級：B（只讀摘要；「TE 在 +100/+200ms 延遲完全失敗」等細節只來自二手摘要，未驗證）。

### 15. Rewind-IL
- 引用：Gehan Zheng, Sanjay Seenivasan, Matthew Johnson-Roberson, Weiming Zhi, 2026, "Rewind-IL: Online Failure Detection and State Respawning for Imitation Learning", IEEE RA-L 2026（Early Access, pp. 1–8, DOI 10.1109/LRA.2026.3734897），arXiv:2604.16683。https://arxiv.org/abs/2604.16683
- 一句貢獻：訓練後免訓練的執行期保護：用相鄰重疊 chunk 的零樣本差異度量（split conformal prediction 校準）偵測失敗，失敗時把機器人退回已驗證的安全 checkpoint 並重啟推論。
- 與本專案關係：可借方法（以預測不一致作為觸發訊號）；原文動作是「偵測失敗後回到安全狀態再重啟」，不是單純重規劃，我們沒實作其方法。
- 可借：overlapping-chunk 差異作為觸發訊號。
- 證據等級：B（只讀摘要）。

### 16. AAC
- 引用：Yuanchang Liang, Xiaobo Wang, Kai Wang, Shuo Wang, Xiaojiang Peng, Haoyu Chen, David Kim Huat Chua, Prahlad Vadakkepat, 2026, "Adaptive Action Chunking at Inference-time for Vision-Language-Action Models", CVPR 2026, arXiv:2604.04161（v2）。https://arxiv.org/abs/2604.04161
- 一句貢獻：固定 chunk 長度在反應性與平順性間取捨，AAC 用動作熵由當前預測決定 chunk 長度；大 chunk 反應差，小 chunk 易 mode-jumping 與 chunk 間不連續。
- 與本專案關係：背景；動態決定 n 的想法（我們的失明感知規則也是動態 n）。假設觀測可靠（推論，摘要未談故障）。它把小 chunk 的壞處歸因於 mode-jumping 而非停頓。
- 可借：動態 n 的討論對照。
- 證據等級：B（只讀摘要）。

### 17. BID
- 引用：Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, Chelsea Finn, ICLR 2025, "Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling", arXiv:2408.17355（v4）。https://arxiv.org/abs/2408.17355 ；專案頁 https://bid-robot.github.io/
- 一句貢獻：chunking 提升時序結構捕捉但降低對意外狀態的反應；BID 測試時抽樣多個候選，以與過去決策的向後一致性及與未來計畫的前向對比選擇動作（7 個模擬基準＋2 個真機）。
- 與本專案關係：支持「n 是連貫與反應的取捨」；與觀測損壞無關。
- 可借：取捨論述的引用。
- 證據等級：B（只讀摘要＋專案頁）。

### 18. Targeted Modality Dropout
- 引用：Genki Shikada, Kazuki Osamura, Masaru Ide, Tetsuya Ogata, Kanata Suzuki, 2026, "Targeted Modality Dropout for Real-Robot Manipulation Robust to Intermittent Vision Loss", arXiv:2610.09566（2026-10-07）。https://arxiv.org/abs/2610.09566
- 一句貢獻：用注意力估計各模態依賴度，有選擇地丟掉最主導的模態（通常是視覺），加依賴分布的熵正則；真機雙臂操作視覺喪失時基線大幅下降而 TMD 維持任務。
- 與本專案關係：重訓路線的對照（我們免訓練偵測 vs 他們重訓）；隨機選模態丟棄的一般 dropout 連無視覺喪失時也在很多任務失敗；沒有 black vs freeze、沒有重規劃機制分析。不確定其策略架構是否為 ACT。
- 可借：discussion 中「training-time 視覺 dropout」類別的代表。
- 證據等級：B（只讀摘要）。

### 19. FIPER（含 Sentinel、FAIL-Detect 同類）
- 引用：Ralf Römer, Adrian Kobras, Luca Worbis, Angela P. Schoellig, 2025, "Failure Prediction at Runtime for Generative Robot Policies" (FIPER), arXiv:2510.09459。https://arxiv.org/abs/2510.09459
- 一句貢獻：不需失效資料；策略 embedding 空間的 random network distillation 偵測 OOD 觀測，加 action-chunk entropy，兩者同時超過閾值才警報，閾值用成功 rollout 的 conformal prediction 校準。
- 與本專案關係：只偵測、不處理偵測後怎麼做；黑影像／凍結幀應被 OOD 觀測訊號抓到（推論），可當比「全黑／與前幀相同」更通用的偵測器對照。同類：Sentinel（Agia 2024，2410.04640，一致性＋進度偵測）與 FAIL-Detect（Xu 2025，RSS 2025，2503.08558，序列 OOD＋conformal），見「其他相關」。
- 可借：「OOD 觀測且高不確定」才觸發；conformal 校準閾值，可為「與前一幀相同」類偵測器設定誤報率。
- 證據等級：B（只讀摘要）。

## 其他相關（一行）
- LIBERO-PRO（Zhou 等，2025，2510.03827，https://arxiv.org/abs/2510.03827）：標準 LIBERO 超過 90% 的模型在擾動設定下降到 0.0%，指出模型靠記憶動作序列；用於記憶化排除檢查。B。
- LIBERO-Plus（Fei 等，2025，2510.13626，https://arxiv.org/abs/2510.13626）：七個擾動維度含感測器雜訊，相機視角與初始狀態最敏感；沒有失明或凍結。B。
- RoboCasa-Shift（Wang 等，2026，2610.27861，https://arxiv.org/abs/2610.27861）：匹配的乾淨與擾動 rollout、分階段診斷、三個嚴重度。B。
- THE COLOSSEUM（Pumacay 等，2024，2402.08191，https://arxiv.org/abs/2402.08191）：14 個環境擾動軸，模擬與真實相關性 R²=0.614；無相機失效。B。
- VLA-GAUGE（2610.16567，https://arxiv.org/abs/2610.16567）：單因子敏感度加組合擾動篩選，12 個 VLA；與相機失效無直接關係。B。
- Sentinel（Agia 等，2024，2410.04640，https://arxiv.org/abs/2410.04640）：動作一致性（erratic）加 VLM 任務進度偵測，多偵出 18% 失效；偵測策略自身行為失效而非輸入端壞掉。B。
- FAIL-Detect（Xu 等，RSS 2025，2503.08558，https://arxiv.org/abs/2503.08558）：失效偵測表述為序列 OOD 偵測，conformal 提供統計保證。B。
- PIER（Li 等，2026，2610.12123，https://arxiv.org/abs/2610.12123）：證據閘門執行介面，涵蓋 stale observation；僅合成軌跡評估、作者自己聲明物理任務表現未確立，可信度低。B。
- The Unexpired Plan（2610.05747，https://arxiv.org/abs/2610.05747）：摘要談監控器是兩件事（讀什麼統計量、觸發後做什麼），並指 chunked 策略上一個計畫未過期可免費讀取；與「觸發後的反應決定效果」同構，值得細讀，場景是加速擴散策略。B。
- Progress Probes（2610.31613）、GUARD（2608.04510）：VLA／擴散策略 runtime monitor。B。
- RL4IL（Ismkhan, Bouchahcia, 2026，2606.15514）：LIBERO 感測器 dropout，檢索示範加軟融合補缺失 embedding，免重訓；與 MINT 同類。B。
- Online Sensor Diagnosis and Repair（Ziegler 等，2026，2610.36852）：外部透明代理診斷並修復凍結四足行走策略的壞輸入；非影像。B。
- Skand 等，CoRL 2024（PMLR 270, pp. 4463–4482，https://proceedings.mlr.press/v270/skand25a.html）：masked training 對感測器失效穩健（RL）；arXiv 編號未找到。B。
- Bohez 等，2017，Sensor Fusion for Robot Control through Deep RL（1703.04550，https://arxiv.org/abs/1703.04550）：對執行期感測器失效穩健的多感測融合架構（摘要未明說 dropout 訓練）。B。
- Vogt-Lowell 等，2026，When Sensors Fail（2603.04648，https://arxiv.org/abs/2603.04648）：持續感測失效下的序列模型 PPO，獎勵退化上界與失效持續時間有關；可借「失效持續時間」變數。B。
- Heess 等，2015，Memory-based control with RNNs（1512.04455，https://arxiv.org/abs/1512.04455）：部分可觀測控制背景。B。
- Sinopoli 等，2004，Kalman Filtering With Intermittent Observations（IEEE TAC，https://doi.org/10.1109/tac.2004.834121）：量測缺失時只預測不更新；可類比為「失明期間不重規劃」的動機（推論）。C（描述來自背景知識，未讀原文）。
- Mixed-Gain Adaption-Based Fault-Tolerant Funnel Control of Robotic Manipulators（IEEE T-ASE 2024，https://doi.org/10.1109/tase.2024.3388592）：低維關節感測器模型式容錯控制，與影像失效關係遠。C。
- Robustness of Robotic Manipulation: Foundations and Frontiers（Dong 等，2026，2606.31494，https://arxiv.org/abs/2606.31494）：穩健性綜述，可當 related work 總引用。B。
- Goyal 2026，Measuring the Stability Assumption Behind Action Chunking（2610.01626，https://arxiv.org/abs/2610.01626）：注入小動作誤差量 open-loop 與 closed-loop 誤差增長，replanning 很少把放大變收斂；擾動注入量測協定可借。B。
- Zhang 等，2025，Action Chunking and Exploratory Data Collection Yield Exponential Improvements in BC（2507.09061，https://arxiv.org/abs/2507.09061）：理論，控制穩定性是 chunking 好處的機制。B。
- 自適應 horizon 家族（共同點：觸發訊號來自策略內部、預測本身或世界模型；皆只讀摘要，B）：DEHP（Zhao 2026，2606.11408）；DVAC（Feng 2026，2606.03847）；Knowing When to Stop（Xu 2026，2609.00908）；BCP（Xu 2026，2608.03483）；A3 Dynamic Execution Commitment（Chen 2026，2605.11567）；CA3C（Li 2026，2610.07949）；Mixture of Horizons（Jing 2025，2511.19433）；A2C2（Sendai 2025，2509.23224）；Temporal Action Selection（Weng 2025/2026，2511.04421）。連結格式皆為 https://arxiv.org/abs/<編號>。
- 檢索中出現但與本專案較遠或未深讀（未列入）：PATCH（2606.16690）、DREAM-Chunk（2606.18589）、Spatial Attention、ChunkTrust、Staircase、FASTER、RobustVLA（2511.01331，Jacobian 正則化 RL 後訓練，針對觀測雜訊）、NeurIPS 2025 "From Faults to Features"（偏車輛動力學）。

## 建議的 baseline／資料集／評估協定
| 項目 | 來源 | 備註 |
|---|---|---|
| 預訓練 ACT n=100 無 temporal ensembling | LeRobot 模型卡 https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human ；config.json（chunk_size 100、n_action_steps 100、temporal_ensembling_coeff null） | 模型卡 83.0%（500 集，80k steps）對原 ACT repo 訓練版 68.0%；模型卡未明說評估 n，但依預設 config 推論對應 n=100。我們測得 n=100 為 80/100、TE 為 83/100；83% 與 83/100 容易讀混，報告需強調 83% 出處是模型卡且對應 n=100，二項標準誤約 4 個百分點（推論），差異在抽樣誤差內 |
| ACT 論文數字 | ACT Table I（2304.13705） | Cube Transfer 人類示範 50%、腳本 86%；不可把 83% 歸給 ACT 論文；模型卡兩數字都高於論文 50% 的原因未查，勿臆測 |
| Temporal ensembling（ACT 原版） | ACT；Median TE 2609.27167 | 失明下的聚合對照；n=1 時等價每步重規劃並平均 |
| Median TE | 2609.27167 | 免訓練穩健聚合；與「不重規劃」比較 |
| 撤回／缺視角（MINT）、一致性縮短（CARE） | 2609.35003；2609.32550 | 概念對照；MINT 需重訓或補視角，多半無法在預訓練 ACT 上直接複現 |
| 失明偵測器：全黑（均值<1.0）、與前幀相同 | Suh 2026 的故障定義；本專案 | 可與 FIPER／FAIL-Detect 的 OOD＋conformal 偵測、CARE 的一致性偵測並列 |
| 條件成功率協定 | MAIL-Bench 2609.35003 | 以策略自己的成功 rollout 為參考，在其完成時間 30–60% 注入，只在完整相機成功的場景上算 |
| 擾動分類與發生階段 | LIBERO-VPro 2609.24350 | blackout／雜訊／常數影像／過期重播；freeze 對應 Camera Staleness |
| 配對 rollout 與階段診斷 | RoboCasa-Shift 2610.27861 | 乾淨與擾動配對 |
| 閾值校準 | FAIL-Detect 2503.08558；FIPER 2510.09459 | conformal 設定偵測器誤報率 |
| 資料集 | 無需資料集 | 用 LeRobot 預訓練權重＋gym-aloha；查無 ALOHA 或 robomimic 專用的相機失效 benchmark |

## 投稿場合
來源：4.md（2026-11-03 查，涵蓋 2026-11 至 2027-07；只採官方頁，查不到寫「未公布」，「二手」僅供參考）。作品特徵：單任務、單模型、合成失明、純模擬、無真機，統計扎實（12,100 次）。頂會主會要進，大概要補真機、第二任務或第二個模型（4.md 作者判斷，不是官方要求）。

| 場合 | 截稿 | 連結 | 適合度 |
|---|---|---|---|
| IROS 2027 主會（佛羅倫斯，2027-09-27 至 10-01） | 2027-03-01（IEEE RAS 活動頁，紐約時間 23:59）；IROS 官網尚未列截稿與頁數 | https://www.ieee-ras.org/event/call-for-papers-paper-submission-deadline-iros-2027-ieee-rsj-international-conference-on-intelligent-robots-and-systems-iros-27403-0/ ；https://2027.ieee-iros.org | 中：截稿最早最近，門檻比 RSS、CoRL 寬；補第二任務或更多模型後可試 |
| CoRL 2027 主會 | 未公布；去年（CoRL 2026）論文截稿 2026-05-28，初投 8 頁，要求 Limitations 節 | https://sites.google.com/corl.org/corl2026/contributions/instruction-for-authors | 中：主題最對口，且不明文要求真機；預期 2027-05 截稿（推測），是「補完實驗再投」的最佳目標 |
| RSS 2027 主會（雅典，2027-07-06 至 11） | 摘要 2026-12-04（AoE）；全文 2027-04-16（AoE）；補充材料 2027-04-19 | https://roboticsconference.org/ ；去年頁 https://roboticsconference.org/2025/information/cfp/ | 低至中：時程可行，但錄取門檻高，單任務純模擬很難過；須有 Limitations 節；本屆頁數未載 |
| ICRA 2027 主會（首爾） | 2026-09-16 23:59 PST（已過）；下一屆 ICRA 2028 預期 2027-09（推測，未查官方頁） | https://2027.ieee-icra.org/call-for-technical-papers/ （見 https://2027.ieee-icra.org/announcements/call-for-technical-papers/ ） | 低：已截稿 |
| ICRA 2027 workshop "Robust Robot Learning for Real-World Deployment" | 官網寫 TBA；2–6 頁，依 ICRA'27 格式 | https://icra27rrl.github.io/ | 高（主題）：題目就是學習策略穩健性、失敗偵測、評估基準；頁面主辦人欄是佔位文字，ICRA 要到 2027-01-31 才通知 workshop 錄取，是否已被接受無法確認，需盯住頁面；未說是否接受純模擬 |
| ICRA 2027 其他 workshop | 各自訂，預期 2027-03 至 04（推測，依去年同類 workshop 截稿 2026-04-15、2026-04-22） | 列表待錄取後出現；去年列表 https://2026.ieee-icra.org/?p=860 ；入口 https://2027.ieee-icra.org/contribute/ | 高：去年有多個對口 workshop（Manipulation Robustness、Uncertainty in Open-World Robotics 等），純模擬短論文最現實 |
| RSS 2027 workshop | 未公布，官網目前沒有 workshop 頁 | https://roboticsconference.org/ | 中：之後再查 |
| CoRL 2027 workshop | 未公布；去年提案 2026-04-29 公布徵求，多半落在 2027-08 至 09，超出本窗口後段 | https://2026.corl.org/contributions/call-for-workshops | 中：可先排進計畫；CoRL 2026 不鼓勵 workshop 收已被主會錄取的論文 |
| IROS 2027 workshop | 未公布；IROS 2026 各 workshop 約 2026-08 截稿（二手） | https://www.ieee-ras.org/events/category/deadlines/ | 低：預期 2027-08，超出窗口 |
| TAAI 2026（新北，2026-11-20 至 21） | 已過：CFP 頁國際與國內軌 2026-09-28 AoE（首頁寫 09-14，兩頁不一致，以 CFP 頁為準）；國內軌全文 6 頁或延伸摘要 2 頁（中英文皆可） | https://www.taai.org.tw/ ；https://taai2026.github.io/zh/cfp.html | 低（今年已過）；TAAI 2027 官方頁尚無資訊，預期 2027-09（推測），有 Best Student Paper 獎但頁面沒寫資格 |
| CVGIP 2027 | 未公布；二手：CVGIP 2022 截稿 2022-07-01；可能在 2027-06 至 07（推測） | 去年主辦頁 https://cvgip2026.stust.edu.tw/ | 低至中：主題偏影像，機器人策略的視覺失明有點關聯但不是核心；勿與蘇州的 cvgip.net 混淆 |
| IEEE Region 10 Student Research Paper Contest | 2027 未公布；2024 年版 2024-05-15 | https://sac.ieeer10.org/srp-contest/ | 中：最多 6 頁，作者皆為學生，第一作者須為 IEEE 學生會員；有 Instrumentation, AI, and Robotics 類別；新版是否辦待查，是否限大學部頁面沒寫 |
| arXiv（cs.RO 或 cs.LG） | 隨時 | https://info.arxiv.org/help/endorsement.html | 高（存檔與引用用）：首次投稿需 endorsement；只是預印本，各會議對先放 arXiv 的規定自行核對（未查） |

推薦順序（來源檔建議）：CoRL 2027 主會（先補實驗）、IROS 2027 主會（2027-03-01）、ICRA／RSS 2027 對口 workshop；備援 arXiv 與 RSS 2027（摘要 2026-12-04、全文 2027-04-16）。查無或無官方來源：全國計算機會議（NCS）、中華民國人工智慧學會相關競賽，不納入。日期衝突：ICRA 2027 截稿官方頁同時出現 09-15 與 09-16，以延長後 09-16 為準。

## 若要投稿要補什麼
**以下全部是依文獻推得的建議（推論），不是任何場合的官方要求。**
1. **第二任務或第二個模型**：來源檔判斷頂會主會至少要補一項（如 gym-aloha 另一個任務、或另一個 chunking 策略如 Diffusion Policy）。目前單任務單模型也是對 Suh 的 VLA 結果方向相反時最需要的佐證。
2. **LIBERO-VPro 式擾動**：加入 Camera Staleness（過期重播，延遲幅度當參數）與暫態 blackout（Temporary Observation Corruption，可控制發生頻率與操作階段），並把「發生階段」當自變數；用其措辭對應我們的 freeze。
3. **MAIL-Bench 式條件成功率協定**：以策略自己的成功 rollout 為參考、在完成時間 30–60% 注入、只在完整相機成功的場景上算條件成功率，降低變異並便於與 MINT 對照；失效分角色（若有多相機）。
4. **與 Median TE、CARE 的對照**：Median TE 作為「聚合而非不重規劃」的免訓練對照，包含稀疏壞幀（q/M）與連續失明兩種設定；CARE 式一致性偵測作為第三種偵測器，並做失明長度掃描，找最佳策略從縮短 n 轉為保持不重規劃的交叉點；FIPER 式 OOD 偵測可與簡單影像偵測並列。
5. **記憶化排除檢查**：LIBERO-PRO 指出標準 benchmark 上模型可能靠記憶動作序列；我們有「失明後仍有成功率」的 baseline，需排除「全黑影像仍成功是因為記憶化軌跡而非視覺使用」，例如擾動初始方塊位置後再測失明。
6. 其他：讀 ActFovea 全文釐清與本體感覺觸發結果是否衝突；補一個 PACE 式低速點重規劃對照以檢驗「n=1 失敗來自 chunk 開頭停頓」；補 Limitations 節（RSS、CoRL 要求）；若時程允許，真機為加分項。

## 檢索紀錄
合併四份來源檔。查無僅代表此次檢索範圍內未找到。

| 日期 | 工具 | 查詢 | 結果數 | 備註 |
|---|---|---|---|---|
| 2026-11-03 | arXiv API（1.md） | id_list=2609.39145,2604.04161,2604.16683,2512.07697,2506.07339,2408.17355 | 6 | 全部命中 |
| 2026-11-03 | WebFetch arXiv abs／HTML（1.md） | 2609.39145（abs 1 次、HTML 3 次，全文 136,008 字元）；2408.17355v1；2304.13705 abs 與 pdf（pdftotext 抽 Table I） | — | 確認 Suh 細節、BID v1 標題、ACT Table I |
| 2026-11-03 | WebFetch HF（1.md） | lerobot/act_aloha_sim_transfer_cube_human 模型卡；config.json（curl） | 2 | 83% 出處與 chunk_size／n_action_steps |
| 2026-11-03 | WebFetch（1.md） | bid-robot.github.io | 1 | 確認 ICLR 2025 與現行標題 |
| 2026-11-03 | OpenAlex（1.md） | BID 標題 1 次（5 筆）；Suh、Rewind-IL 各 1 次 | — | 後兩者回應被截斷，未取得可用結果，不採用 |
| 2026-11-03 | openreview 搜尋 API（1.md） | BID | 0 | 404 |
| 2026-11-03 | arXiv API（2.md） | `all:"action chunking" AND all:"execution horizon"`；`all:"action chunk" AND all:replanning`；`all:"temporal ensembling" AND all:action`；`all:"adaptive" AND all:"action chunk"`；`all:"closed-loop" AND all:"action chunking"`；`all:"open-loop" AND all:"action chunk"`；`all:"action chunk" AND (blank OR occlusion OR "camera failure" OR "sensor failure")` | 15／15／11／15／15／15／12 | sortBy=relevance；之後 9 個查詢（camera dropout、pause、chunk boundary、chunk size ablation、temporal ensembling latency、hesitation 等）遇 HTTP 429，結果無效未計入；id_list 最後一批 429，改讀 arxiv.org/abs |
| 2026-11-03 | OpenAlex（2.md） | action chunking execution horizon；action chunk replanning frequency；temporal ensembling action chunking；adaptive action chunk size；closed-loop action chunking robustness（from 2023-01-01） | 各 10 | 用於交叉確認標題；多數被引數為 0 |
| 2026-11-03 | WebSearch（2.md） | chunk 開頭停頓／小 horizon 失敗；ACT 相機失效／黑畫面；觀測 dropout 與 action chunking | 3 次 | 只作線索，結論回到一手頁面 |
| 2026-11-03 | WebFetch（2.md） | arXiv abs：2606.16690、2608.02547、2511.04421、2606.18589、2609.27167、2607.29169、2506.07339、2408.17355；HTML：2609.27167、2608.15938（前 100k 字元）；PDF：ACT、Diffusion Policy（pdftotext） | — | 第 4–12 項大多僅讀摘要 |
| 2026-11-03 | arXiv API（3.md） | camera failure policy；sensor dropout imitation；visuomotor robustness occlusion；failure detection diffusion policy；observation corruption robot policy；Blackout Freeze VLA；robustness VLA perturbation；LIBERO-PRO；LIBERO-Plus；action chunking real-time latency；temporal ensembling robustness | 成功（筆數未記） | 另有多個 AND 查詢（camera fault robot policy；runtime monitor imitation policy；missing modalities manipulation；modality dropout manipulation；camera occlusion imitation；robomimic perturbation benchmark；ALOHA robustness；observation delay imitation；frozen frame policy 等）遇 429 回空，並非確認查無；後續 ti: 查詢（DisDP、masked training sensor failure）也 429 |
| 2026-11-03 | OpenAlex（3.md） | visuomotor policy camera failure robustness；sensor dropout imitation learning；robot policy occlusion robustness benchmark；failure detection diffusion policy runtime monitoring；observation corruption VLA robustness；RL sensor failure observation dropout；RL missing observations dropout；ModDrop modality dropout；sensor fault tolerant control manipulator；RL random delays；camera failure blackout frozen；imitation missing camera view；VLA camera view dropout；Kalman intermittent observations；Heess memory-based control；Colosseum（查無）；Decomposing generalization gap；Targeted modality dropout；Latent Policy Barrier；VLA-GAUGE；RoboCasa-Shift（查無，但 abs 頁可讀）；SAFE multitask failure detection；robustness benchmark image corruption | — | 429 後的替代來源 |
| 2026-11-03 | WebFetch arXiv abs／HTML（3.md） | abs：2609.35003、2610.12123、2609.39145、2609.32550、2606.15514、2410.04640、2503.08558、2510.13626、2609.24350、2506.07339、2610.31613、2603.04648、1703.04550、2510.03827、2610.05747、2608.04510、2609.27167、2511.01331、2510.09459、2610.09566、2606.31494、2610.27861、1512.04455、2402.08191、2610.16567、2610.36852；HTML 全文抽查：2609.35003、2609.24350、2609.32550、2609.39145、2609.27167 | 26 | 另 WebSearch／WebFetch 讀 Skand 等 CoRL 2024 PMLR 頁面 |
| 2026-11-03 | WebFetch／WebSearch（4.md） | 官方頁：ICRA 2027（首頁、contribute、technical papers）；RSS 首頁與 RSS 2025 CFP；CoRL 首頁、CoRL 2026 instruction for authors、CoRL 2026 與 2025 call for workshops；IEEE RAS deadlines 與 IROS 2027 活動頁、IROS 2027 官網；TAAI 官方首頁與 TAAI 2026 CFP；arXiv endorsement；IEEE R10 SAC srp-contest；ICRA 2026 workshop 列表；icra27rrl.github.io | — | 失敗或無資訊：iros.org（憑證錯誤）；iros2027.org、www.ippr.org.tw、www.cvgip.org.tw（網域不存在）；roboticsconference.org/2027/information/cfp/（404）；CVGIP 2025／2026 頁無截稿日；nstc.gov.tw 首頁無大專學生計畫公告。二手：學校公告、各 IROS 2026 workshop 頁經搜尋轉述、mldeadlines／PaperPilot 彙整站（未採用為截稿依據） |

Semantic Scholar 全程未使用。
**限流註記**：arXiv API 在 2.md 與 3.md 的檢索中途被 HTTP 429 限流，後半段多個查詢回傳空結果，不代表真的零筆；改用 OpenAlex、WebSearch 與直接讀 abs 頁補查。因此本文所有「查無」只代表此次檢索範圍內未找到，不保證不存在。
未查證：ICLR 2025 會議頁、RSS 2023 頁面、DA-DP／AAC／Rewind-IL／RTC 全文（只讀摘要）；DisDP（Vanjani 等，RL Journal 2025，MINT 引用為缺失視角訓練的代表）未讀到；Skand 等 CoRL 2024 的 arXiv 編號。

## 未解項
- [NEEDS CLARIFICATION: ActFovea（2607.29169）全文——其觸發條件（本體感覺一致性）是否與我們「本體感覺觸發無效」結果衝突]
- ~~Median TE 原文核對~~：已於 2026-11-04 結案，見「要注意的衝突」。
- [NEEDS CLARIFICATION: Suh Table 5 數字對照原 PDF（目前來自摘要器引文）]
- [NEEDS CLARIFICATION: Suh 與我們 black／freeze 方向相反的原因假設（凍結畫面時機、ACT 對黑幀 OOD）尚未驗證]
- [NEEDS CLARIFICATION: 預訓練 ACT n=1 失敗是否來自 chunk 開頭停頓——查無直接文獻，需自行量測]
- [NEEDS CLARIFICATION: ICRA 2027 "Robust Robot Learning for Real-World Deployment" workshop 是否已被錄取與其截稿日]
- [NEEDS CLARIFICATION: CoRL／RSS／IROS 2027 workshop 與 CoRL 2027、TAAI 2027、CVGIP 2027 官方日期]
- [NEEDS CLARIFICATION: ACT 模型卡 83%、68% 皆高於 ACT 論文 50% 的原因（訓練資料量、評估協定、LeRobot 版環境）未查，勿在論文臆測]
- [NEEDS CLARIFICATION: DisDP 與 Skand 等 CoRL 2024 的 arXiv 編號尚未取得]

修正紀錄 2026-11-04：Median TE（2609.27167）的描述有三處互相不一致（摘要把空白幀實驗寫成「robomimic」而未指明策略與任務、「含 ACT」的空白對凍結比較、條目 3 把 ACT 0.010→0.595 當作空白幀結果且 ACT／Diffusion Policy 乾淨成功率 0.85／1.00 被當成衝突），已對照原文統一：空白幀實驗只在 Diffusion Policy、robomimic can、一個 stride（h=2）；ACT 的 0.595 是對抗性攻擊；0.85 是 ACT、1.00 是 Diffusion Policy，均為 h=2。
