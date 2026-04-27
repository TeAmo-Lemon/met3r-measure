
# MEt3R（中文说明）

MEt3R 是一个用于评估两视角图像一致性的可微指标（CVPR 2025）。本仓库基于官方实现，并补充了本地离线评测脚本，适合批量评估重建结果（如 3DGS 输出）。

项目主页：<https://geometric-rl.mpi-inf.mpg.de/met3r/>

## 1. 功能概览

- 支持 `MASt3R` / `DUSt3R` / `RAFT` 三种 warping backbone。
- 支持 `cosine`、`lpips`、`rmse`、`psnr`、`mse`、`ssim` 等距离度量。
- 提供离线优先的单目录评测脚本 [mytest.py](mytest.py)：
  - 输入一个 `renders` 文件夹
  - 单目录模式通过 `--frame-step` 控制帧间隔
  - 带进度条
  - 输出标准化结果行：`RESULT pairs=... mean_score=...`
  - 支持批量模式：`--batch-root-dir` 自动遍历 `*/train/ours_*/renders`
  - 批量模式支持 short/long 双步长：`--short-step` 与 `--long-step`
  - 批量模式每完成一个目录即写入一行 `tsv`（实时落盘）
- 提供批量评测脚本 [test_batch.sh](test_batch.sh)：
  - 自动遍历 `*/train/ours_*/renders`
  - 同时计算 short(`step=1`) 与 long(`step=5`)
  - 结果保存为 `tsv`
  - 支持增量复用：已成功目录自动跳过，避免重复计算

## 2. 环境要求

- Python >= 3.10（推荐）
- PyTorch >= 2.1.0
- CUDA >= 11.3（GPU 评测建议）
- PyTorch3D
- 其余依赖见 [requirements.txt](requirements.txt)

## 3. 安装步骤

在仓库根目录执行：

```bash
pip install -r requirements.txt
git submodule update --init --recursive
pip install -e . --no-build-isolation
```

### 准备本地 FeatUp（离线模式）

`mytest.py` 默认从本地目录加载 FeatUp：`third_party/FeatUp`。

```bash
mkdir -p third_party
git clone https://github.com/mhamilton723/FeatUp.git third_party/FeatUp
```

首次运行时如需缓存权重，请在可联网环境先跑一次；后续可离线复用本机缓存。

## 4. 单目录评测（mytest.py）

示例（short-range，一帧间隔）：

```bash
python mytest.py \
  --input-dir /path/to/renders \
  --frame-step 1
```

示例（long-range，五帧间隔）：

```bash
python mytest.py \
  --input-dir /path/to/renders \
  --frame-step 5
```

可选参数（精简保留）：

- `--pairing adjacent|all_to_first`（默认 `adjacent`）
- `--backbone raft|mast3r|dust3r`（默认 `raft`）
- `--distance cosine|lpips|rmse|psnr|mse|ssim`（默认 `cosine`）
- `--img-size`（默认 `256`）

输出示例：

```text
RESULT pairs=160 mean_score=0.184309
```

## 5. 批量评测（mytest.py）

`mytest.py` 支持直接批量遍历目录并输出 `tsv`：

```bash
python mytest.py \
  --batch-root-dir /mnt/data2/experiments/3dgs/output_Stylized \
  --short-step 1 \
  --long-step 5
```

批量模式逻辑：

- 搜索匹配 `*/train/ours_*/renders` 的目录
- 每个目录执行两次评测：
  - short: `frame_step=--short-step`
  - long: `frame_step=--long-step`
- 每个目录评测完成后，立即追加写入一行 `met3r_batch_results_YYYYMMDD_HHMMSS.tsv`
- 排序为自然顺序（示例：`test` → `test1` → `test2` → `test10` → `test50` → `test100`）
- 模型仅加载一次，在整个批任务中复用，任务结束后再释放缓存

输出列定义（mytest 批量模式）：

```text
dataset	short_step	short_pairs	short_mean_score	short_status	long_step	long_pairs	long_mean_score	long_status	renders_dir	message
```

## 6. 批量评测（test_batch.sh）

默认遍历目录：`/mnt/data2/experiments/3dgs/output_Stylized`

```bash
bash test_batch.sh
```

或指定根目录：

```bash
bash test_batch.sh /mnt/data2/experiments/3dgs/output_Stylized
```

脚本逻辑：

- 对每个 `renders` 目录计算两组指标：
  - short: `frame_step=1`
  - long: `frame_step=5`
- 解析 `mytest.py` 输出的 `RESULT` 行
- 保存结果到 `met3r_batch_results_YYYYMMDD_HHMMSS.tsv`
- 默认增量模式：自动复用最近一次已成功结果

若需强制全量重算：

```bash
FORCE_RECALC=1 bash test_batch.sh
```

## 7. 结果文件格式（TSV）

列定义：

```text
dataset	short_step	short_pairs	short_mean_score	short_status	long_step	long_pairs	long_mean_score	long_status	renders_dir
```

注：`mytest.py --batch-root-dir` 生成的 `tsv` 会额外包含 `message` 列，用于记录失败原因。

## 8. 常见问题

### 8.1 为什么会访问网络？

- 如果使用 `mhamilton723/FeatUp` 这类字符串，`torch.hub` 会按远程仓库处理。
- 当前脚本已改为默认本地 FeatUp 路径，优先离线。

### 8.2 出现 `NA` / `failed` 怎么办？

- 常见原因：网络瞬时波动、单目录异常、手动中断。
- 重新执行 `bash test_batch.sh` 即可增量补算失败项（不会重复全部目录）。

## 9. 参考与引用

如果你的工作使用了 MEt3R，请引用原论文：

```bibtex
@inproceedings{asim25met3r,
  title     = {MEt3R: Measuring Multi-View Consistency in Generated Images},
  author    = {Asim, Mohammad and Wewer, Christopher and Wimmer, Thomas and Schiele, Bernt and Lenssen, Jan Eric},
  booktitle = {IEEE/CVF Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```
