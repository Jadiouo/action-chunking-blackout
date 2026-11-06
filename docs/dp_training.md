# Diffusion Policy 訓練設定（P4 第一段）

目的：在 `lerobot/aloha_sim_transfer_cube_human`（50 條人類示範、20000 幀、50 Hz、單相機 `top` 480x640、狀態與動作皆 14 維）上用 LeRobot 0.4.4 訓練 Diffusion Policy（DP），作為 ACT 之外的第二種動作分塊策略，之後在失明網格上比較。

## 入口與檔案
- 訓練器：`.venv/bin/lerobot-train`（0.4.4 的 `lerobot.scripts.lerobot_train`），DP 為 `--policy.type=diffusion`。
- `code/dp_train.sh <run> <steps> <batch>`：包 lerobot-train（固定超參數；若 `runs/dp/<run>/checkpoints/last` 已存在則自動 `--resume=true` 續跑）。
- `code/dp_run.sh <run> <total_steps> <batch>`：一個 gpujob 內含停損閘（見下）＋續跑＋單工作 5.67h 時限（到時自動停，重送同一指令即續跑）＋VRAM 記錄（`runs/dp/<run>_vram.log`）。
- `code/dp_eval.py`：評估（見「評估」）。
- 輸出：`runs/dp/<run>/`（已加進 `.gitignore`，不 commit 權重）；資料與模型下載在 `.hf/`。
- 執行（本機 gpujob）：
  `gpujob systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 bash code/dp_run.sh dp_v1 60000 64`
  （環境變數 OMP/MKL threads=2、MUJOCO_GL=egl、HF_HOME、HF_HUB_DISABLE_IMPLICIT_TOKEN 已在腳本內設定。）

## 超參數與出處
出處代號：[LR] = lerobot 0.4.4 `DiffusionConfig` 預設值（`.venv/lib/python3.10/site-packages/lerobot/policies/diffusion/configuration_diffusion.py`，其預設對應 PushT）；[DP-real] = 原作者 repo `real-stanford/diffusion_policy` 的 `diffusion_policy/config/train_diffusion_unet_real_hybrid_workspace.yaml`（真實機器人、影像 240x320 的設定，2026-11-03 取自 raw.githubusercontent.com）；[DP-sim] = 同 repo `train_diffusion_unet_hybrid_workspace.yaml`（PushT 影像）。

| 超參數 | 值 | 出處 / 理由 |
|---|---|---|
| n_obs_steps | 2 | [LR]=[DP-real]=[DP-sim] |
| horizon（預測長度） | 64 | 我們自訂。DP 預設 16 步 @10 Hz（1.6 s），執行一半（8）。ALOHA 為 50 Hz，64 步約 1.28 s；也需 ≥ 後續網格 n∈{8,16,32} 加 n_obs−1，且須為 8 的倍數（3 層下採樣，[LR] 的檢查） |
| n_action_steps（DP 預設執行步數） | 32 | 我們自訂：沿用 DP 的「執行 horizon 的一半」比例（8/16），horizon 64 → 32。評估時另補 n=8、16 供之後網格參考 |
| drop_n_last_frames | 31 | [LR] 公式 horizon − n_action_steps − n_obs_steps + 1 = 64−32−2+1 |
| 影像處理 | 先縮放 240x320，再隨機裁 216x288（crop_ratio 0.9；評估為中心裁） | [DP-real] 影像 240x320、`crop_shape [216,288]`；ALOHA 原圖 480x640 縮半後與其相同 |
| vision backbone | resnet18、GroupNorm、無預訓練、SpatialSoftmax 32 關鍵點、單相機單編碼器 | [LR]=[DP-real]（hybrid） |
| down_dims | (256, 512, 1024) | [DP-real]（[LR] 預設 512/1024/2048 對應 PushT 版 [DP-sim]）。較小、較快；參數量 76M |
| kernel_size / n_groups / step-embed dim / FiLM scale | 5 / 8 / 128 / True | [LR]=[DP-real] |
| 擴散排程 | num_train_timesteps=100、beta 0.0001→0.02、squaredcos_cap_v2、epsilon 預測、clip_sample ±1 | [LR]=[DP-real]（`DDIM` 排程，訓練噪音相同） |
| 推論取樣 | DDIM、8 步 | [DP-real]（`num_inference_steps: 8`）。[LR] 預設為 DDPM 100 步，慢 12 倍；單次規劃約 8 次 U-Net 前向 |
| 正規化 | 影像 mean/std；狀態與動作 min-max 到 [−1,1] | [LR] 預設；由資料集統計量計算並存進 checkpoint 的 processor 檔 |
| 優化器 | Adam，lr 1e-4，betas (0.95, 0.999)，eps 1e-8，weight decay 1e-6，grad clip 10 | [LR]=[DP-real] |
| LR 排程 | cosine，warmup 500 步 | [LR]=[DP-real] |
| batch size | 64 | [DP-real]；VRAM 見下 |
| 訓練步數 | 60000（≈192 epoch，312 步/epoch） | 我們自訂，由 GPU 預算反推（見下）；[DP-real] 為 600 epoch，[DP-sim] 3050 epoch，預算不允許 |
| EMA | 無 | 偏離原作：lerobot 0.4.4 的 DP 沒有 EMA（原作 `use_ema: True`）。由 cosine 衰減到 0 的學習率部分補償 |
| DataLoader | num_workers=4、seed=1000 | 專案限制 num_workers ≤4；seed 為 lerobot 預設 |
| 資料增強 | 只有隨機裁切（無 color jitter） | [LR] 預設 `image_transforms` 關閉 |
| 影片解碼 | lerobot 預設（torchcodec，AV1） | — |
| checkpoint | 每 2500 步；訓練前 20000 步為停損閘 | 我們自訂 |

## 預算與停損（主 AI session 指示）
- 單一 gpujob ≤6h、VRAM ≤10GB、本段訓練總 GPU ≤12h 硬上限（目標 ≤10h）。
- 速度試跑（bs64、200 步，本機被其他程序佔用 CPU 的情況下）：見「結果」。
- 停損閘：訓練到 20000 步時（步 20000 的 checkpoint 一寫完就停）先評估 20 seeds（0–19）；成功率 <10% 即停止、不續訓、回報主 AI session。實作在 `code/dp_run.sh`，結果寫 `results/dp_gate_dp_v1_summary.json`。
- 不無限續訓：60000 步結束後若無失明成功率 <30%，在此檔記錄並評估是否續訓（再 ≤6h），由主 AI session 決定。

## 評估
`code/dp_eval.py`：`gym_aloha/AlohaTransferCube-v0`、`obs_type=pixels_agent_pos`、400 步、成功 = 最大獎勵 4（與 `run_grid.py` 相同）；env seed = seed0+episode；用 checkpoint 內的 processor 做正規化；規劃 = 動作佇列為空時（每 n 步）；Wilson 95% CI。擴散取樣噪音以 `torch.manual_seed(env_seed)` 固定（但整條管線並非位元級確定，見 `docs/rulers.md`）。沿用 `code/blackout.py` 的失明介面（`--L/--mode`，預設 L=0），但**尚未**接上 `run_t3t4.py` 的 oracle/detect 迴圈（後半段）。注意 DP 有 n_obs_steps=2 的觀測佇列，失明時佇列中會混入黑畫面/凍結畫面。

## 第三方 checkpoint `sangil109/diffusion_aloha_sim`（只讀 config，未執行任何其程式碼）
- Hub 上只有 `config.json`、`train_config.json`、`model.safetensors`，**沒有 model card**，也沒有宣稱成功率。更新日期 2025-06-20。
- 資料集：`lerobot/aloha_sim_transfer_cube_human`（即 Transfer Cube）。
- 設定幾乎全是 lerobot 預設（PushT 風格）：horizon 16、n_action_steps 8、down_dims 512/1024/2048、DDPM 100 步、batch 8、100000 步、**影像隨機裁成 84x84（從 480x640 原圖，不縮放）**、`output_dir=outputs/train/aloha_sim_test`、`job_name=aloha_sim_diffusion`；訓練期 eval 50 episode 每 20000 步（結果未附）。
- 判斷：裁 84x84 小塊代表每個畫面只看到全圖約 2.5%，方塊與夾爪多半不在視野內，成功率很可能很低；舊格式（使用 `use_amp`，無 processor 檔），在 0.4.4 載入要自行補正規化統計量。無任何宣稱數字，不能當成「已訓練好的 DP」捷徑；僅可當作低成本的弱對照（評估 50 episode 約 20 分鐘），是否採用由主 AI session 決定。

## 結果

### 速度與資源（bs64）
- 速度：約 3.6–3.7 步/秒（0.27 s/步；其中資料載入等待約 0.1–0.16 s，CPU 被其他程序占用）。60000 步預估 4.5–5h。
- VRAM：整張卡最高 8289 MiB（含顯示用約 1.3GB），本工作約 7.0GB，在 ≤10GB 內。
- 主機記憶體：cgroup（MemoryMax=12G）使用量最高 10.5GB（匿名 5.0GB、檔案快取約 4–5GB、共享記憶體約 3.5GB，後者來自 DataLoader worker）。
- 實際 GPU 用量：速度試跑約 1.5 分鐘；第一次正式跑（#470）在第 2601 步（約 11 分鐘）被 12G 上限 OOM 砍掉（journal：`A process of this unit has been killed by the OOM killer`），已保留 2500 步 checkpoint；第二次（#476，從 2500 步續跑，另關閉 DataLoader 的 pin_memory）跑到 20000 步＋閘門評估，共 7775 s（2.16h）。合計約 2.4 GPU 小時，遠低於 12h 上限。

### 停損閘：20000 步，無失明，n_action_steps=32，20 seeds（0–19）
- 成功 0/20，Wilson 95% CI [0, 0.161]；最大獎勵全為 0（連方塊都沒碰到）。`results/dp_gate_dp_v1.csv`、`results/dp_gate_dp_v1_summary.json`。
- 依預先規則（<10% 即停）**已停止訓練、不續訓**；60000 步與 50 seeds 的正式評估（n=32/16/8）都**沒有做**。
- 訓練 loss（epsilon 預測 MSE）0.006（第 20000 步）；checkpoint：`runs/dp/dp_v1/checkpoints/020000/pretrained_model`（不 commit）。

### 評估管線檢查（CPU，離線）
為排除評估程式出錯：取 4 個訓練幀，用同一 checkpoint 與資料集統計量手動正規化，比較預測的 32 步動作與示範動作。平均絕對誤差 0.012–0.057（幀 10：0.057；3000：0.012；9000：0.041；15000：0.019；單位為關節弧度／夾爪行程），與該段示範動作自身的起伏（0.006–0.029）同量級或更大；夾爪開合時機大致對（例如 3000 幀預測 1.03 對示範 1.04）。環境觀測與資料集幀的格式一致（影像 float32 [0,1]、3x480x640、平均亮度 0.156 對 0.155；狀態順序相同）。所以管線大致正確，低成功率來自模型尚未精確（20k 步＝64 epoch，且無 EMA、8 步 DDIM 取樣）。這個檢查是 4 幀的粗略 sanity check，不是完整驗證。

### 診斷（2026-11-04，主 AI session 追加；全在 CPU 上跑，GPU 用量 0）
停損閘 0/20 且最大獎勵全為 0（連方塊都沒碰到），先排除管線問題，再試取樣與執行步數。
1. 執行步數：n_action_steps=8，seeds 0–19：0/20，Wilson 95% CI [0, 0.161]，最大獎勵全 0（`results/dp_diag_n8.csv`）。
2. 取樣器（同 checkpoint、n=8、seeds 0–9）：DDIM 50 步 0/10、DDPM 100 步 0/10，各 CI [0, 0.278]，最大獎勵全 0（`results/dp_diag_n8_ddim50.csv`、`dp_diag_n8_ddpm100.csv`；程式 `code/dp_eval.py --infer ddim50|ddpm100`）。
3. 示範動作直接在 env 重放（資料集第 0、1、2 集，env seed 0–3，開環，`.scratch/dp_diag/replay.py`）：第 0、1 集在 seed 0、1、3 成功（最大獎勵 4），seed 2 沒成功；第 2 集四個 seed 都沒成功。失敗者推測是方塊位置與示範不同或示範本身不穩（沒有逐一查證）。重點是動作空間對得上：重放時 env 關節狀態與示範狀態的平均絕對差約 0.002。
4. 開環逐幀餵第 0 集觀測（`.scratch/dp_diag/openloop_ep0.png`）：預測 chunk 的前約 16 步貼著示範，之後出現「提前進入下一階段」的偏離（例如右前臂在第 50 幀預測就往 −0.3 掉，示範要到約第 210 幀才掉）。這是時機的多峰性：示範開頭靜止很久、何時出手不固定。
5. 閉環軌跡（`.scratch/dp_diag/rollout_s0.png`）：政策很快偏離示範的相位（左臂在示範該靜止時抬起、右前臂與手腕提早旋轉），之後整個動作序列錯亂。
6. 三項檢查，皆無 bug：
   - 反正規化有套用：checkpoint 內 processor 的 min/max 與資料集統計量相同（動作 min/max 對得上），預測動作落在正確的物理範圍（夾爪約 0.2–1.1，關節 ±1.2 以內）。
   - 影像：env 第 0 幀與資料集第 0 幀同為 RGB、float [0,1]、CHW，平均絕對差 0.42/255，差異像素 0.02%（只有方塊位置不同），每通道平均亮度相同。
   - 觀測佇列：`policy.reset()` 後觀測與動作佇列長度皆為 0；`dp_eval.py` 每個 episode 開頭都呼叫 reset。
7. 結論：**沒有找到管線 bug**。模型本身在閉環中偏離示範：20k 步（64 epoch，無 EMA，lerobot 預設 PushT 風格設定）下，僅 50 條示範、示範開頭時機不確定，政策提早切換階段，閉環誤差累積。我們沒有驗證「訓練更久就會成功」。

### 結論與處置（P4 第一段）
- DP（本設定，20k 步 checkpoint）無失明成功率 0%（n=32：0/20；n=8：0/20；DDIM 50 步 n=8：0/10；DDPM 100 步 n=8：0/10）。依 P4 驗證條款（<30% 照實記錄並縮減範圍）與停損規則，**P4 標為「負結果、縮減」**：不再續訓，也不做 DP 的失明網格。
- 若日後要再試，需要先決定是否投入新的訓練預算（≥數 epoch 倍、EMA、較小 horizon 等），目前沒有證據顯示會成功。

(以下為停損閘當下的建議，已被上面的診斷取代。)
- 目前證據（0/20）不支持續訓到 60k。若仍想嘗試 DP，較可能有用但**尚未驗證**的方向：(a) 評估 n=8/16（較短的開環片段）看是否只是 n=32 太長；(b) 取樣改 DDPM 100 步；(c) 加 EMA 或更多 epoch（需自訂訓練迴圈／patch lerobot）。每一項都要再花 GPU 時數（各 ≤6h 的量級），超出本段已用的 2.4h 之外。
- P4 驗證條款「若 DP 無失明成功率 <30% 照實記錄並縮減範圍」：目前為 0/20（20k 步），依此縮減。

## 第二輪（2026-11-04，再試一次 DP；v1 的結論見上）
預算：訓練 ≤10h GPU、單一工作 ≤6h；若成功，失明網格另 ≤4h。

### v2：同一份 human 資料，改 EMA 與 DP 原作的 sim 設定
| 超參數 | v2 值 | 與 v1 的差異／出處 |
|---|---|---|
| horizon / n_obs_steps / n_action_steps | 16 / 2 / 8 | [DP-sim]=[LR] 預設（DP 原作 sim 任務）；v1 為 64/2/32（我們自訂） |
| drop_n_last_frames | 7 | 公式 16−8−2+1 |
| EMA | decay 0.999，warm-up min(0.999,(1+t)/(10+t))；評估一律用 EMA 權重 | 新增。原作 `use_ema: True`（power 0.75、max 0.9999）；0.999 為主 AI session 指定。實作在 `code/dp_train_main.py`（lerobot 的 `policy.update()` 掛鉤），權重存 `pretrained_model/ema.safetensors` |
| 影像 | 縮放 240x320、隨機裁 216x288 | 同 v1 |
| 其餘（down_dims 256/512/1024、DDIM 8 步、lr 1e-4、bs 64、cosine、warmup 500） | 同 v1 | 同 v1 |
| 總步數 | 名目 100000（約 7.7h）；40k 步為閘門 | 我們自訂；實際 GPU 時數以 10h 為上限 |
| save_freq | 10000 | 磁碟空間有限（v1 為 2500）。代價：被打斷時最多損失 10000 步（約 45 分鐘） |
- 閘門：到 40000 步評估 20 seeds，≥10% 才續訓；未過改做 v3。中途（10k、20k、30k）用 CPU 評估 n=8、seeds 0–9（`code/dp_cpu_watch.sh`，結果 `results/dp_mid_dp_v2_*`）。
- horizon 16 的限制：n_action_steps ≤ horizon−n_obs+1 = 15，所以 n=16 做不到；最終評估改 n∈{4,8,15} 各 50 seeds，網格用其中最佳的兩個 n（主 AI session 同意）。一次擴散取樣只給 15 步，故 oracle 在 n=15 等同 fixed。
- 啟動指令：`gpujob env DP_HORIZON=16 DP_NACT=8 DP_EMA_DECAY=0.999 systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 bash code/dp_run.sh dp_v2 100000 64 40000`（工作 #522）。

### v2 結果（工作 #522，2026-11-04 17:24–約 20:35，約 3.2 GPU 小時，到 40000 步）
- 速度 3.7 步/秒（資料載入為瓶頸），VRAM 7.8GB（`nvidia-smi` 單程序），無 OOM。
- 中途 CPU 評估（EMA，n=8，seeds 0–9）：10k 0/10、20k 0/10、30k 0/10（`results/dp_mid_dp_v2_*`），最大獎勵全為 0。
- **閘門 40k（EMA，n=8，seeds 0–19）：成功 0/20，Wilson 95% CI [0, 0.161]**；最大獎勵 18 次為 0、2 次為 2（碰到並舉起方塊，但沒完成傳遞），平均 0.2（`results/dp_gate_dp_v2*`）。比 v1（20k：全為 0）有進步，閉環軌跡（`.scratch/dp_diag/rollout_v2_20k.png`、`rollout_v2_40k.png`）顯示右臂會伸向方塊、夾爪在大致正確的時間張開，v1 的「提早進入下一階段」現象減輕。
- 依閘門規則（<10% 停）v2 不續訓，剩餘訓練預算改做 v3。v2 的 040000 checkpoint 保留（`runs/dp/dp_v2/checkpoints/040000`）。

### v3（備案，v2 在 40k 仍 <10% 才做）
資料換成 `lerobot/aloha_sim_transfer_cube_scripted`（腳本示範，出手時機固定、沒有開頭長靜止段），其餘同 v2。**這和 ACT 預訓練 checkpoint 所用的資料不同（ACT 為 human 版），論文須照實註明。**

### v3 結果
- 工作 #565，約 1.4 GPU 小時到 20k 步：閘門 0/20，Wilson 95% CI [0, 0.161]，最大獎勵全為 0（`results/dp_gate_dp_v3*`）。10k 步為 0/10。依規則不續訓。

### v2 延長（最後一次）
- 2026-11-06 決定：v2 從 40k 續訓到 100k，這是它的學習率排程原本設定的步數（`code/dp_v2_extend.sh`）。理由：目的是「加長訓練」；40k 只有 128 epoch，v2 又是目前唯一有進展的版本（2/20 舉起方塊），資料也與 ACT 相同。v3 資料不同，且到 20k 毫無進展，所以不延長。預估約 4.5 GPU 小時，訓練累計約 9h，在 10h 內。
- 停損：100k 步評估 20 seeds（n=8）。<30% 就結束 P4，論文寫成負結果，不再加訓練；≥30% 才評估 n∈{4,8,15} 各 50 seeds，再跑失明網格（≤4h）。

### v2 延長結果（定案：負結果）
- 工作 #1，2026-11-06：100k 步閘門（EMA，n=8，seeds 0–19）成功 2/20，Wilson 95% CI [0.028, 0.301]；最大獎勵 14 次 0、1 次 1、3 次 2、2 次 4，平均 0.75（`results/dp_final_dp_v2_100000*`）。
- 2/20 <30%，依事先寫好的停損規則 P4 結束：不再訓練，不跑失明網格，論文只報 ACT 並照實寫這段負結果。CI 上界剛好到 0.30，不改變決定，因為規則看的是點估計。

### 失明網格用的程式 `code/dp_grid.py`（自己的迴圈，不改 `run_t3t4.py`）
規則 fixed／oracle 的定義同 `run_t3t4.py`，K=horizon−n_obs+1=15 為一次取樣的可用步數：fixed 每 n 步重規劃；oracle 在失明時繼續執行舊 chunk，chunk（15 步）用完才被迫重規劃。DP 特有：觀測佇列（n_obs=2）每個 env 步都收入政策看到的畫面（失明幀也進佇列）。輸出 `results/dp.csv`，可續跑。

## 清理紀錄（磁碟空間有限；訓練期間只留最新 2 份、閘門 040000 與 last 所指者）
- 2026-11-04 已手動清掉 v1 的 checkpoint，只留 runs/dp/dp_v1/checkpoints/020000。
- 2026-11-04 19:49:16 清理 runs/dp/dp_v2/checkpoints/010000（1.2G）；保留：020000 030000 040000 
- 2026-11-04 20:31:41 清理 runs/dp/dp_v2/checkpoints/020000（1.2G）；保留：030000 040000 
- 2026-11-06 05:09:44 清理 runs/dp/dp_v2/checkpoints/030000（1.2G）；保留：040000 090000 100000 
- 2026-11-06 05:09:44 清理 runs/dp/dp_v2/checkpoints/050000（1.2G）；保留：040000 090000 100000 
- 2026-11-06 05:09:44 清理 runs/dp/dp_v2/checkpoints/060000（1.2G）；保留：040000 090000 100000 
- 2026-11-06 05:09:44 清理 runs/dp/dp_v2/checkpoints/070000（1.2G）；保留：040000 090000 100000 
- 2026-11-06 05:09:44 清理 runs/dp/dp_v2/checkpoints/080000（1.2G）；保留：040000 090000 100000 
