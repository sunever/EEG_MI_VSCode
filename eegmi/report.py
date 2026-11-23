"""Summarize out-of-fold predictions, subject-paired effects and EEG explanations."""
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from eegmi.data import CHANNELS, MODEL_HZ, SPECTRAL_BANDS

METRICS = ("balanced_accuracy", "macro_f1", "accuracy")


def metrics(y, p):
    return [balanced_accuracy_score(y, p),
            f1_score(y, p, labels=[0, 1], average="macro", zero_division=0),
            accuracy_score(y, p)]


def score_predictions(predictions):
    rows = []
    for (name, fold, subject), rows_for_subject in predictions.groupby(["model", "fold", "subject"], sort=False):
        rows.append([name, fold, subject, len(rows_for_subject),
                     *metrics(rows_for_subject.label, rows_for_subject.prediction)])
    return pd.DataFrame(rows, columns=["model", "fold", "subject", "n_trials", *METRICS])


def paired_inference(subjects, seed):
    pairs = [("fusion", model) for model in ("wide", "wide_matched", "csp_lda", "fbcsp_lda", "mu", "beta")]
    pairs.append(("weighted", "fusion"))
    rows = []
    rng = np.random.default_rng(seed)
    for metric in METRICS:
        table = subjects.pivot(index="subject", columns="model", values=metric)
        for first, second in pairs:
            if {first, second}.issubset(table.columns):
                delta = (table[first] - table[second]).dropna().to_numpy()
                boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
                p_value = wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0
                rows.append([first, second, metric, len(delta), delta.mean(), low, high, p_value])
    paired = pd.DataFrame(rows, columns=["model_a", "model_b", "metric", "n_subjects",
                                         "mean_difference", "ci95_low", "ci95_high", "wilcoxon_p"])
    paired["p_holm_within_metric"] = np.nan
    for _, group in paired.groupby("metric"):
        ordered = group.wilcoxon_p.sort_values()
        adjusted = np.maximum.accumulate(ordered.to_numpy() * np.arange(len(ordered), 0, -1))
        paired.loc[ordered.index, "p_holm_within_metric"] = np.minimum(1, adjusted)
    return paired


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_main(out, predictions, subject_scores, models, pairs):
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.12), 4.8), layout="constrained")
    arrays = [subject_scores.loc[subject_scores.model == name, "balanced_accuracy"] * 100 for name in models]
    box = ax.boxplot(arrays, tick_labels=models, patch_artist=True, showfliers=False, showmeans=True)
    for artist in box["boxes"]:
        artist.set_facecolor("#cee7e4")
    rng = np.random.default_rng(2026)
    for i, values in enumerate(arrays, 1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(values)), values, s=10, alpha=0.32, color="#17656a")
    ax.axhline(50, color="#a8432f", linestyle="--", linewidth=1, label="50% chance BA")
    ax.set(ylabel="Subject balanced accuracy (%)", ylim=(0, 104), title="Out-of-fold subject performance")
    ax.tick_params(axis="x", labelrotation=23)
    ax.legend(loc="lower right")
    save_figure(fig, out / "subject_distribution.png")

    cols = min(4, len(models))
    fig, axes = plt.subplots(int(np.ceil(len(models) / cols)), cols,
                             figsize=(3.2 * cols, 3.05 * np.ceil(len(models) / cols)), squeeze=False)
    confusions = []
    for ax, name in zip(axes.flat, models):
        selected = predictions[predictions.model == name]
        counts = confusion_matrix(selected.label, selected.prediction, labels=[0, 1])
        rates = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for i, j in np.ndindex(counts.shape):
            ax.text(j, i, f"{counts[i, j]:,}\n{rates[i, j]:.1%}", ha="center", va="center",
                    color="white" if rates[i, j] > 0.57 else "#12324d")
            confusions.append([name, i, j, counts[i, j], rates[i, j]])
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Left", "Right"],
               yticklabels=["Left", "Right"], xlabel="Predicted", ylabel="True", title=name)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, out / "confusion_matrices.png")
    pd.DataFrame(confusions, columns=["model", "true_label", "pred_label", "count",
                                     "row_normalized"]).to_csv(out / "confusion_matrices.csv", index=False)

    table = pairs[pairs.metric == "balanced_accuracy"]
    if not table.empty:
        fig, ax = plt.subplots(figsize=(7.6, 0.50 * len(table) + 1.7), layout="constrained")
        mean = table.mean_difference.to_numpy() * 100
        row = np.arange(len(table))
        bounds = table[["ci95_low", "ci95_high"]].to_numpy() * 100
        ax.hlines(row, bounds[:, 0], bounds[:, 1], color="#1b676d", linewidth=2)
        ax.plot(mean, row, "o", color="#1b676d")
        ax.axvline(0, color="#606060", linestyle="--", linewidth=1)
        ax.set(yticks=row, yticklabels=(table.model_a + " − " + table.model_b),
               xlabel="Paired mean BA difference (percentage points)", title="Subject bootstrap intervals (95%)")
        ax.invert_yaxis()
        save_figure(fig, out / "paired_balanced_accuracy.png")


def montage_info():
    info = mne.create_info(list(CHANNELS), MODEL_HZ, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    return info


def plot_maps(out, values, titles, filename, unit, low, high, cmap):
    ncols = min(len(values), 3)
    fig, axes = plt.subplots(int(np.ceil(len(values) / ncols)), ncols, squeeze=False,
                             figsize=(3.1 * ncols + .7, 3.0 * np.ceil(len(values) / ncols)),
                             layout="constrained")
    for ax, table, title in zip(axes.flat, values, titles):
        image, _ = mne.viz.plot_topomap(table, montage_info(), axes=ax, show=False,
                                       vlim=(low, high), contours=4, cmap=cmap)
        ax.set_title(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), label=unit, shrink=0.74)
    save_figure(fig, out / filename)


def summarize_masks(out, predictions, subject_scores, masks, seed):
    baseline = predictions[predictions.model == "fusion"].set_index("trial_id").loc[masks.index]
    baseline_subject = subject_scores[subject_scores.model == "fusion"].set_index("subject")
    rows = []
    for subject, data in baseline.groupby("subject"):
        clean = baseline_subject.loc[subject, "balanced_accuracy"]
        for condition in masks.columns:
            p = masks.loc[data.index, condition].to_numpy() > 0.5
            ba = balanced_accuracy_score(data.label, p)
            rows.append([subject, data.fold.iloc[0], condition, clean, ba, clean - ba])
    scores = pd.DataFrame(rows, columns=["subject", "fold", "condition", "baseline_ba",
                                        "masked_ba", "ba_drop"])
    scores.to_csv(out / "occlusion_subject_scores.csv", index=False)
    rng = np.random.default_rng(seed)
    rows = []
    for name, group in scores.groupby("condition", sort=False):
        values = group.ba_drop.to_numpy()
        sampled = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(sampled, [0.025, 0.975])
        rows.append([name, len(values), values.mean(), low, high])
    summary = pd.DataFrame(rows, columns=["condition", "n_subjects", "mean_ba_drop", "ci95_low", "ci95_high"])
    summary.to_csv(out / "occlusion_summary.csv", index=False)
    effects = summary.set_index("condition").loc[[f"channel_{channel}" for channel in CHANNELS]]
    values = effects.mean_ba_drop.to_numpy() * 100
    span = max(abs(values).max(), 0.01)
    plot_maps(out, [values], ["Single-channel masking"], "channel_occlusion_topomap.png",
              "Baseline − masked BA (pp)", -span, span, "RdBu_r")
    groups = summary[~summary.condition.str.startswith("channel_")]
    mean = groups.mean_ba_drop.to_numpy() * 100
    row = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.6, 5), layout="constrained")
    ax.hlines(row, groups.ci95_low * 100, groups.ci95_high * 100, color="#1b676d", linewidth=2)
    ax.plot(mean, row, "o", color="#1b676d")
    ax.axvline(0, color="#606060", linestyle="--", linewidth=1)
    ax.set(yticks=row, yticklabels=groups.condition, xlabel="Baseline − masked BA (pp)",
           title="Spectral and spatial masking on held-out subjects")
    ax.invert_yaxis()
    save_figure(fig, out / "group_occlusion.png")
    return groups


def describe_spectra(out, epochs):
    shape = epochs.power.shape
    channel_columns = pd.MultiIndex.from_product([SPECTRAL_BANDS, CHANNELS],
                                                  names=["band", "channel"])
    power = 10 * np.log10(np.maximum(epochs.power, 1e-12))
    subject_label = [epochs.trials.subject, epochs.trials.label]
    p = pd.DataFrame(power.reshape(shape[0], -1), columns=channel_columns).groupby(subject_label).mean()
    e = pd.DataFrame(epochs.erd.reshape(shape[0], -1), columns=channel_columns).groupby(subject_label).mean()
    p = p.stack(["band", "channel"], future_stack=True).rename("imagery_power_db").reset_index()
    e = e.stack(["band", "channel"], future_stack=True).rename("erd_db").reset_index()
    per_subject = p.merge(e, how="left", on=["subject", "label", "band", "channel"])
    per_subject.to_csv(out / "spectral_subject_channels.csv", index=False)
    summary = per_subject.groupby(["label", "band", "channel"], sort=False).agg(
        imagery_power_db=("imagery_power_db", "mean"),
        erd_db=("erd_db", "mean"), n_erd_subjects=("erd_db", "count"),
    ).reset_index()
    summary.to_csv(out / "spectral_channel_summary.csv", index=False)
    titles = [f"{hand} | {band}" for hand in ("Left", "Right") for band in SPECTRAL_BANDS]
    table = summary.set_index(["label", "band", "channel"]).sort_index()
    for measure, filename, label, cmap in (
        ("imagery_power_db", "band_power_topomaps.png", "Imagery power [dB re 1 µV²]", "viridis"),
        ("erd_db", "erd_topomaps.png", "Imagery/rest power ratio (dB)", "RdBu_r"),
    ):
        values = [table.loc[(hand, band)].loc[list(CHANNELS), measure].to_numpy()
                  for hand in (0, 1) for band in SPECTRAL_BANDS]
        if all(np.isfinite(value).all() for value in values):
            if measure == "erd_db":
                high = max(np.abs(values).max(), .01)
                low = -high
            else:
                low, high = np.min(values), np.max(values)
                high = max(high, low + .01)
            plot_maps(out, values, titles, filename, label, low, high, cmap)
    return int(np.isfinite(epochs.erd).all(axis=(1, 2)).sum())


def generate_markdown(out, study, epochs, scores, folds, fits, pairs, masks, erd_count):
    n = scores.subject.nunique()
    labels = epochs.trials.label.value_counts()
    qc = pd.read_csv(out / "qc.csv")
    excluded = qc.loc[qc.status == "excluded_subject", "subject"].tolist()
    rows = ["# 跨被试运动想象识别实验结果", "",
            f"运行配置为 **{study['config']['profile']}**，设备为 `{study['device']}`，"
            f"随机种子为 {study['config']['seed']}。测试覆盖 **{n} 名被试、"
            f"{len(epochs.trials):,} 个试次**（左手 {labels[0]:,}、右手 {labels[1]:,}）。"
            f"其中 {erd_count:,} 个试次有可用的同次试验前 T0 基线。",
            "",
            "每名被试仅出现在一个外层测试折。被试级指标先在同一被试内计算，再对被试等权平均。"
            "下表的折间标准差取各折被试等权平均后的样本标准差。训练用时为各折拟合与选参耗时的中位数。"
            "传统模型的试次协方差在计时前逐频段计算。",
            ""]
    if study["config"]["profile"] == "quick":
        rows.extend(["**快速模式只用于验证流程。**", ""])
    if excluded:
        rows.extend([f"质控后没有可用于二分类的数据的被试为 {', '.join(excluded)}。"
                     "被试、记录与试次的审计见 [qc.csv](qc.csv)。", ""])
    rows.extend(["| 模型 | 被试均值 BA | 被试均值 Macro-F1 | 被试均值 Accuracy | 折间 BA 均值 ± SD | 参数量 | 每折耗时中位数 |",
                 "|---|---:|---:|---:|---:|---:|---:|"])
    for name in study["config"]["models"]:
        s = scores[scores.model == name][list(METRICS)].mean()
        f = folds[folds.model == name].balanced_accuracy
        model_fits = fits[fits.model == name]
        parameters = model_fits.trainable_parameters.dropna()
        count = f"{int(parameters.iloc[0]):,}" if len(parameters) else "见模型统计表"
        formatted = " | ".join(f"{s[metric]:.2%}" for metric in METRICS)
        rows.append(f"| {name} | {formatted} | {f.mean():.2%} ± {f.std()*100:.2f} pp | {count} | "
                    f"{model_fits.train_seconds.median():.1f} s |")
    rows.extend(["", "![各被试 Balanced Accuracy](subject_distribution.png)", "",
                 "![合并外层预测的混淆矩阵](confusion_matrices.png)", "",
                 "## 配对差值与频段融合", ""])
    show = pairs[pairs.metric == "balanced_accuracy"]
    if len(show):
        main = show[(show.model_a == "fusion") & (show.model_b == "wide")]
        if len(main):
            effect = main.iloc[0]
            if effect.ci95_low > 0:
                interpretation = "区间整体为正，本次观察支持拼接融合在该划分下提高 BA。"
            elif effect.ci95_high < 0:
                interpretation = "区间整体为负，本次观察支持宽频基线在该划分下取得更高 BA。"
            else:
                interpretation = "区间跨零，本次比较尚未给出融合能稳定提高 BA 的证据。"
            rows.extend([f"**主要对照。** 拼接融合相对宽频 EEGNet 的被试平均 BA 差值为 "
                         f"{effect.mean_difference*100:+.2f} 个百分点。{interpretation}", ""])
        matched = show[(show.model_a == "fusion") & (show.model_b == "wide_matched")]
        if len(matched):
            effect = matched.iloc[0]
            rows.extend([f"**参数量对照。** 拼接融合相对加宽宽频模型的 BA 差值为 "
                         f"{effect.mean_difference*100:+.2f} 个百分点，95% 区间为 "
                         f"[{effect.ci95_low*100:+.2f}, {effect.ci95_high*100:+.2f}] 个百分点。", ""])
        rows.extend(["配对检验的单位为被试。以下区间为 10,000 次被试重抽样的 95% 百分位区间。"
                     "外层各折共享部分训练被试，统计结果用于描述本次协议下的配对差异。",
                     "", "| 对照（前者减后者） | 差值（pp） | 95% CI（pp） | Holm 校正 p |",
                     "|---|---:|---:|---:|"])
        for effect in show.itertuples():
            rows.append(f"| {effect.model_a} − {effect.model_b} | {effect.mean_difference*100:+.2f} | "
                        f"[{effect.ci95_low*100:+.2f}, {effect.ci95_high*100:+.2f}] | "
                        f"{effect.p_holm_within_metric:.4g} |")
        rows.extend(["", "![BA 配对差值](paired_balanced_accuracy.png)", ""])
    rows.extend(["## 头皮分布与遮蔽", ""])
    if masks is not None:
        rows.append("在各外层折的测试数据上，将标准化后的对应频段或通道置零，保持其它输入与模型权重固定。"
                    "BA 降幅按被试配对计算，正值表示遮蔽后 BA 下降。"
                    "地形图描述模型对电极信号的依赖，遮蔽会改变输入分布。")
        rows.extend(["", "| 遮蔽条件 | 平均 BA 降幅（pp） | 95% CI（pp） |",
                     "|---|---:|---:|"])
        for effect in masks.itertuples():
            rows.append(f"| {effect.condition} | {effect.mean_ba_drop*100:+.2f} | "
                        f"[{effect.ci95_low*100:+.2f}, {effect.ci95_high*100:+.2f}] |")
        rows.extend(["", "![分组遮蔽](group_occlusion.png)", "",
                     "![单通道遮蔽头皮分布](channel_occlusion_topomap.png)", ""])
    rows.append("频带功率来自事件后 [0.5, 2.5) 秒。ERD/ERS 用同试次的事件前 [−2.25, −0.25) 秒 T0 功率作参考。"
                "负 dB 表示功率降低，正 dB 表示功率升高。图中各类别先在被试内平均，再对被试等权平均。"
                "运动节律、视觉提示和记录噪声都可能影响头皮分布。")
    for file, title in (("band_power_topomaps.png", "想象期频段功率"), ("erd_topomaps.png", "ERD/ERS")):
        if (out / file).exists():
            rows.extend(["", f"![{title}]({file})", ""])
    rows.extend(["", "## 复核文件", "",
                 "- [模型统计与选择结果](model_fits.csv)",
                 "- [每个被试的指标](subject_metrics.csv)",
                 "- [每折指标](fold_metrics.csv)与[折间均值和标准差](fold_metric_summary.csv)",
                 "- [全部测试预测](predictions.csv)与[配对统计](paired_comparisons.csv)",
                 "- [运行环境和被试划分](study.json)与[可用试次表](trials.csv)",
                 "- [频带功率与 ERD 数表](spectral_channel_summary.csv)", ""])
    (out / "report.md").write_text("\n".join(rows), encoding="utf-8")


def build_report(out, epochs, study):
    predictions, fits, masks = [], [], []
    models = study["config"]["models"]
    for split in study["splits"]:
        fold = split["fold"]
        for name in models:
            path = out / "folds" / f"fold_{fold:02d}" / name
            p = pd.read_csv(path / "predictions.csv")
            if sorted(p.subject.unique()) != split["test"]:
                raise ValueError(f"Fold {fold}, {name}: prediction subjects differ from the test split.")
            predictions.append(p.assign(fold=fold, model=name))
            fits.append(json.loads((path / "fit.json").read_text(encoding="utf-8")))
            if name == "fusion":
                masks.append(pd.read_csv(path / "mask_probabilities.csv", index_col="trial_id"))
    predictions = pd.concat(predictions, ignore_index=True)
    fits = pd.DataFrame(fits)
    if predictions.duplicated(["model", "trial_id"]).any() or not all(
        predictions.groupby("model").size() == len(epochs.trials)
    ):
        raise ValueError("Report requires one held-out prediction per eligible trial and per model.")
    scores = score_predictions(predictions)
    folds = scores.groupby(["model", "fold"], sort=False).agg(
        n_subjects=("subject", "size"), n_trials=("n_trials", "sum"),
        **{metric: (metric, "mean") for metric in METRICS},
    ).reset_index()
    seed = study["config"]["seed"]
    pairs = paired_inference(scores, seed)
    for frame, filename in ((predictions, "predictions.csv"), (fits, "model_fits.csv"),
                            (scores, "subject_metrics.csv"), (folds, "fold_metrics.csv"),
                            (pairs, "paired_comparisons.csv")):
        frame.to_csv(out / filename, index=False)
    folds.groupby("model")[list(METRICS)].agg(["mean", "std"]).to_csv(out / "fold_metric_summary.csv")
    plot_main(out, predictions, scores, models, pairs)
    condition_summary = summarize_masks(out, predictions, scores, pd.concat(masks), seed) if masks else None
    erd_count = describe_spectra(out, epochs)
    generate_markdown(out, study, epochs, scores, folds, fits, pairs, condition_summary, erd_count)
