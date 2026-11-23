# 基于多频段脑电特征融合的跨被试运动想象识别

用 PhysioNet EEGMMIDB 研究一个问题：在目标被试零校准条件下，显式融合 μ 和 β 子频段能否提高左右手运动想象识别。工程覆盖数据核验与预处理、五折跨被试训练、模型和消融实验、被试配对统计、频带功率与头皮地形图。

## 1. 下载与打开工程

1. 将 `EEG_MI_VSCode_Full.zip` 解压到一个新目录，例如 `D:\course`。完整版附带正式实验使用的全部 327 个 EDF。若只想快速验证，可以下载并解压 `EEG_MI_VSCode_Quick.zip`。快速版附带 S001–S008 的 24 个 EDF 和相同的实验代码，可执行 `download_data.py` 补齐数据。
2. 打开 VS Code，选择 **文件 → 打开文件夹 → EEG_MI_VSCode**，并安装扩展 **Python**（Microsoft，ID 为 `ms-python.python`）和 **Python Debugger**（用于 F5）。确认 `run.py` 与 `data/` 在同一级。
3. 通过 **终端 → 新建终端** 打开项目终端。Windows 以下使用 PowerShell。

解压后的关键路径：

```text
EEG_MI_VSCode/
├── README.md
├── run.py                         实验入口与折划分
├── download_data.py               原始数据下载、修复、校验
├── requirements.txt
├── .vscode/                       解释器推荐与 F5 运行配置
├── eegmi/
│   ├── data.py                    EDF 读取、质控、分段、频段与功率
│   ├── models.py                  CSP、FBCSP、EEGNet 及各个变体
│   └── report.py                  统计、地形图、结果文档
├── data/
│   ├── README.md                  数据来源与许可
│   ├── LICENSE-ODC-BY.txt
│   ├── SHA256SUMS.txt
│   └── raw/S001/S001R04.edf       示例路径
└── reference_results/             实际运行的参考结果
```

正式运行会创建 `data/cache/` 和 `outputs/full/`。建议预留至少 6 GB 可用磁盘空间及 8 GB 内存。纯 CPU 可运行。多折并发会增加内存占用。

## 2. Windows 环境配置

安装 **64 位 Python 3.12**。安装后重开 VS Code 和终端，然后运行：

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -c "import sys, mne, torch, sklearn; print(sys.executable); print(mne.__version__, torch.__version__, sklearn.__version__)"
```

按 **Ctrl+Shift+P → Python: Select Interpreter**，选择工程中的 `.venv\Scripts\python.exe`。若列表没有该项，用 **Enter interpreter path** 手动指定。上面的命令直接调用虚拟环境，无需激活，也无需修改 PowerShell 执行策略。

若出现 **No suitable Python runtime found**，运行 `py -0p` 或 `py list` 查找已安装的解释器。若需要安装，从 [Python 官方下载页](https://www.python.org/downloads/) 安装 Python 3.12，重启 VS Code 后再运行 `py -3.12 --version`。若 `python --version` 已显示 3.12 且 `python` 可用，可以执行 `python -m venv .venv` 创建环境。 [VS Code Python 官方指南](https://code.visualstudio.com/docs/python/python-quick-start)

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Linux CPU 安装：

```bash
.venv/bin/python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
```

Apple Silicon macOS 安装：

```bash
.venv/bin/python -m pip install "torch==2.8.0"
```

在 VS Code 选择 `.venv/bin/python`。以下运行步骤中的 Windows 解释器路径，在 macOS/Linux 下替换为 `.venv/bin/python` 即可。若要用 NVIDIA GPU，请安装与你的驱动匹配的 PyTorch 2.8.0 CUDA 构建，然后以 `--device cuda --jobs 1` 运行。 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)

## 3. 验证数据并运行

### 快速检查

完整或快速版解压后可直接执行：

```powershell
.\.venv\Scripts\python.exe download_data.py --subjects 1-8 --verify-only
.\.venv\Scripts\python.exe run.py --profile quick
```

此模式使用 S001–S008、三折外层评估、每个深度模型最多 2 个训练周期。它运行全部模型、绘图和报告流程。完成后在 VS Code 打开 `outputs/quick/report.md`，按 **Ctrl+Shift+V** 预览。也可通过 **运行和调试 → 快速验证（8 名被试）→ F5** 启动；若停在断点，按 F5 继续。

### 正式五折实验

安装依赖并解压完整版后执行：

```powershell
.\.venv\Scripts\python.exe download_data.py --verify-only
.\.venv\Scripts\python.exe run.py --profile full
```

若使用快速版，下面的命令从官方公开数据源补齐缺失文件，也可修复损坏文件。已通过 SHA-256 的文件会复用，下载中断后重复执行即可。

```powershell
.\.venv\Scripts\python.exe download_data.py
```

需要分步运行时，可先构建缓存、再并行训练，或在所有折结束后重新生成报告：

```powershell
.\.venv\Scripts\python.exe run.py --profile full --prepare-only
.\.venv\Scripts\python.exe run.py --profile full --jobs 2 --threads 2
.\.venv\Scripts\python.exe run.py --profile full --report-only
```

`--jobs` 是并发的外层折数，`--threads` 是每折使用的 CPU 线程数。单个测试折可用 `--fold 1` 至 `--fold 5` 逐次运行，使用相同配置和输出目录即可。作业中断后，重复同一命令会跳过已经成功保存的折和模型。修改被试范围、模型列表、设备或周期上限时，用 `--out outputs/另一个名称` 指定独立输出目录。例如：

```powershell
.\.venv\Scripts\python.exe run.py --profile full --models csp_lda wide fusion --out outputs/main_only
```

## 4. 数据处理与划分

- **数据版本。** PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0。下载与校验范围为 109 人 × 3 个任务运行（R04、R08、R12）。T1 对应想象左拳动作（标签 0），T2 对应想象右拳动作（标签 1）。T0 休息段仅用于 ERD 参考。
- **窗口及质控。** MNE 读取 EDF+ 中的内嵌注释，统一 64 个通道的命名和顺序。以事件起点截取半开区间 `[0, 4)` 秒。根据实际持续时间、下一个注释和文件边界检查重叠与完整性。排除采样率偏离 160 Hz 或通道集合不匹配的记录，以及非有限值、单通道峰峰值超过 5,000 µV 或超过 8 个通道峰峰值低于 0.5 µV 的试次。进入评估的被试须有两类有效试次。
- **已核验数据。** S088、S092、S100 的 9 个记录原始采样率为 128 Hz，按统一协议排除。S104 的 R08 有一个持续 2.975 秒的 T1 试次，未满 4 秒。共保留 106 名被试的 4,767 个试次，其中左手 2,405、右手 2,362。完整数据归档保留了原始的 327 个记录。详见 `qc.csv`。
- **滤波与标准化。** 每个 4 秒试次分别进行四阶 Butterworth SOS 前后向带通过滤，再通过多相抗混叠重采样到 80 Hz，得到 64 × 320 的矩阵。每折每频段逐通道标准化参数仅用训练被试估计。离线零相位滤波使用完整的 4 秒窗口。
- **受试者隔离。** 在可用被试 ID 上，以 seed 2026 打乱后做外层 5 折。每折从其余被试中用 seed + 折号抽取约 20% 作为验证被试。全部模型共享同一划分。测试被试不参与标准化拟合、CSP、LDA、网络参数估计、学习率选取或早停。每个被试只产生一次外层测试预测。

对筛选出的 EDF 原始文件和数据许可有任何疑问，可查看 [数据来源与引用](data/README.md)。

## 5. 模型与消融

| 命令名 | 输入 | 角色 |
|---|---|---|
| `csp_lda` | 8–30 Hz | 宽频 CSP + 收缩 LDA |
| `fbcsp_lda` | 8–13、13–20、20–30 Hz | 各频段独立 CSP，拼接对数方差特征后做收缩 LDA |
| `wide` | 8–30 Hz | 宽频 EEGNet 基线 |
| `mu` | 8–13 Hz | μ 单频段对照 |
| `beta` | 13–30 Hz | β 单频段对照 |
| `fusion` | 8–13、13–20、20–30 Hz | 三个独立编码分支，拼接后分类 |
| `wide_matched` | 8–30 Hz | 加宽单编码器，接近三分支的参数量 |
| `weighted` | 8–13、13–20、20–30 Hz | 门控输出每试次的分支权重，按权重求和后分类 |

CSP 对训练试次计算迹归一化协方差，使用固定 0.1 的单位矩阵收缩，解广义特征值问题，按特征值两端交替取投影方向。对投影方差取对数，每频段保留 4、6、8 个分量之一，由验证被试的平均 BA 决定。FBCSP 的各频段特征直接拼接。

EEGNet 采用时间滤波、深度空间滤波和可分离卷积。实现利用线性时间滤波与空间投影可交换的性质，先进行空间投影，每对投影共享一个长度为 33 的时间卷积核，再使用批归一化。这是针对本任务实现的 EEGNet 结构变体。随后为 ELU、平均池化、Dropout 与深度可分离时域卷积。单分支宽度为 4，空间深度为 2。参数对照通过枚举 4–24 的整数宽度选择参数量最接近三分支的宽度。PyTorch 使用 AdamW、交叉熵、批大小 64。正式模式每模型在 0.001、0.003 中选学习率，每个候选最多 12 个周期，4 个周期未改善则早停。每个候选以验证集被试平均 BA 选最佳周期，平分时保留先达到的候选。固定种子用于划分、初始化和批次重排。

## 6. 如何阅读结果

正式训练输出位于 `outputs/full/`。打开 [五折正式实验参考报告](reference_results/full/report.md) 可先浏览已跑出的结果，然后比较自己机器生成的输出。不同硬件上的训练耗时会变化，浮点差异也可能影响少数试次预测。

| 文件 | 用途 |
|---|---|
| `report.md` | 实验结果表、配对差值、图像与解读 |
| `study.json` | 所有折的训练、验证、测试被试，种子与环境版本 |
| `qc.csv`、`trials.csv` | 数据质控与清洗后试次索引 |
| `subject_metrics.csv` | 每模型每被试的 BA、Macro-F1 和 Accuracy |
| `fold_metrics.csv`、`fold_metric_summary.csv` | 各折的被试等权平均及折间均值、样本标准差 |
| `predictions.csv`、`confusion_matrices.csv` | 外层测试试次的左右手预测及混淆矩阵 |
| `model_fits.csv` | 选参结果、时间和参数量。传统模型给出保存的系数数量 |
| `paired_comparisons.csv` | 被试配对差值、bootstrap 区间、Wilcoxon 检验与 Holm 校正 |
| `occlusion_summary.csv` | 频段、通道组和独立通道遮蔽的被试配对 BA 变化 |
| `spectral_channel_summary.csv` | 每类、每个频带、每个通道的想象期功率和 ERD/ERS |
| `folds/fold_*/模型/` | 每折模型权重、试次预测和验证历史 |

**主要评价指标。** Balanced Accuracy（BA）为左、右手召回率的平均值，Macro-F1 为两类 F1 的平均值，Accuracy 为正确率。先按被试分别计算，再进行被试等权平均。BA 的二分类随机参照为约 50%。试次合并的混淆矩阵按真实类别行归一化。

**核心对照。** 先比较 `fusion − wide`，然后检查 `fusion − wide_matched` 与 `fusion − fbcsp_lda`。对同一被试的差值进行 10,000 次被试级 bootstrap，报告 95% 百分位区间。Wilcoxon 为双侧，Holm 校对同一指标的多项模型比较。训练折有共享被试，区间未重拟合模型，适合描述本次划分下的配对变化。结果跨零时可报告现有比较未给出清晰的提升证据。

**通道与频段解释。** 对已训练好的 `fusion`，分别置零标准化后的 μ、β1、β2 分支输入以及电极组或单通道。所有遮蔽用同一外层测试试次和同一模型。左右侧及中线组分别包含 C3、C4、Cz 及附近通道，双侧中央组和枕区参照组均含 9 个通道。正的 BA 降幅表示模型表现下降。单电极地形图呈现遮蔽影响，相关通道可能互相补偿。

**频率功率与 ERD。** 在想象段 `[0.5, 2.5)` 秒及前一 T0 的 `[−2.25, −0.25)` 秒计算 Welch PSD，并对 μ、β1、β2 频段积分。绘图中 ERD/ERS 为 `10 log10(想象段功率 / T0 功率)`。负值对应功率降低。先在同一被试、同一类别内对试次平均，再对被试等权平均。地形图由模型依赖或观察性功率变化得到，允许视觉提示、噪声及多种脑过程共同影响。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| 找不到 `run.py` | 重新打开内层 `EEG_MI_VSCode`。在终端运行 `dir` 或 `ls`，应能看到 `run.py`。 |
| `No module named mne/torch` | 在项目终端用 `.venv` 内的 Python 重跑对应的安装命令。F5 前检查 VS Code 右下角解释器。 |
| `Missing or corrupt ...edf` | 检查 `data/raw/Sxxx/SxxxRyy.edf`。可重解压数据包，或运行 `download_data.py` 自动补齐。 |
| `Output directory contains a different experiment setup` | 使用原来的设备、版本和运行选项继续，或将新实验写到另一个 `--out`。 |
| 内存不足 | 使用 `--jobs 1 --threads 2`。可先跑 `--profile quick` 检查软件环境。 |
| GPU 不可用或报确定性算子错误 | 检查驱动和 PyTorch 构建，或使用默认的 CPU 模式。 |

## 参考文献

- Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset* (version 1.0.0). PhysioNet. [DOI 10.13026/C28G6P](https://doi.org/10.13026/C28G6P)
- Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. [DOI 10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
- Lotte, F., et al. (2018). A review of classification algorithms for EEG-based brain-computer interfaces: A 10 year update. *Journal of Neural Engineering*, 15(3), 031005. [DOI 10.1088/1741-2552/aab2f2](https://doi.org/10.1088/1741-2552/aab2f2)
- Ang, K. K., Chin, Z. Y., Zhang, H., & Guan, C. (2008). Filter bank common spatial pattern (FBCSP) in brain computer interface. *IEEE IJCNN*, 2390–2397. [DOI 10.1109/IJCNN.2008.4634130](https://doi.org/10.1109/IJCNN.2008.4634130)
