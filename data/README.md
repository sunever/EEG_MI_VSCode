# EEGMMIDB 数据来源、版本与许可

原始数据为 Gerwin Schalk 及 Wadsworth Center BCI R&D Program 提供的 PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0（2009）。数据按 **Open Data Commons Attribution License v1.0** 提供。本工程重新打包时保留了所选 EDF+ 文件的原始字节和目录命名，并随包提供官方 SHA-256 清单。

- Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset* (version 1.0.0). PhysioNet. RRID:SCR_007345. [DOI 10.13026/C28G6P](https://doi.org/10.13026/C28G6P)
- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System*. IEEE Transactions on Biomedical Engineering, 51(6), 1034–1043. [DOI 10.1109/TBME.2004.827072](https://doi.org/10.1109/TBME.2004.827072)
- Pollard, T., et al. (2026). *PhysioNet as a global platform for biomedical research*. Nature Health. [DOI 10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z)
- [数据说明及下载入口](https://physionet.org/content/eegmmidb/1.0.0/) · [ODC-By 1.0 许可原文](https://physionet.org/content/eegmmidb/view-license/1.0.0/)

完整任务数据归档包含 `S001` 至 `S109` 的 R04、R08、R12，共 327 个 EDF+。T0/T1/T2 注释已内嵌于 EDF+，官方单独的 `.edf.event` 文件是相同注释的兼容副本。工程主包同时附有 S001–S008 的 24 个 EDF+ 供快速验证。官方仓库还提供执行动作、双手/双脚及静息等其他记录。本文实验的纳入条件限定为单侧手部运动想象三次记录。

归档路径示例为 `EEG_MI_VSCode/data/raw/S001/S001R04.edf`。质控剔除的记录也保留在完整任务数据归档中，可用 `python download_data.py --verify-only` 校验所有 327 个文件。
