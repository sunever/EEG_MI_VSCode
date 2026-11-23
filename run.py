"""Reproduce the cross-subject motor imagery experiments and their reports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory

import mne
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import torch

from download_data import ROOT, parse_subjects
from eegmi.data import CHANNELS, prepare
from eegmi.models import MODELS, TRADITIONAL, train_csp, train_net, trial_covariances

PROFILES = {
    "quick": {"subjects": "1-8", "folds": 3, "epochs": 2, "patience": 2,
              "learning_rates": [0.003], "batch_size": 64, "seed": 2026},
    "full": {"subjects": "1-109", "folds": 5, "epochs": 12, "patience": 4,
             "learning_rates": [0.001, 0.003], "batch_size": 64, "seed": 2026},
}
MASKS = {
    "motor_left": "FC3 C3 CP3 C1 C5",
    "motor_midline": "FCz Cz CPz C1 C2",
    "motor_right": "FC4 C4 CP4 C2 C6",
    "motor_bilateral": "FC3 C3 CP3 FCz Cz CPz FC4 C4 CP4",
    "occipital_control": "PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz",
}
OCCLUSION = {name: [CHANNELS.index(channel) for channel in group.split()]
             for name, group in MASKS.items()}
OCCLUSION.update({f"channel_{channel}": [i] for i, channel in enumerate(CHANNELS)})


def make_folds(subjects, folds, seed):
    subjects = np.asarray(sorted(set(subjects)))
    if folds < 3 or len(subjects) < folds:
        raise ValueError(f"Need at least {folds} eligible subjects and at least three folds.")
    splits = []
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (development, testing) in enumerate(splitter.split(subjects), 1):
        rng = np.random.default_rng(seed + fold)
        order = rng.permutation(subjects[development])
        validation_size = max(1, int(np.ceil(0.2 * len(order))))
        train = sorted(order[validation_size:])
        validation = sorted(order[:validation_size])
        test = sorted(subjects[testing])
        if not train or set(train) & set(validation) or set(train + validation) & set(test):
            raise ValueError(f"Invalid subject assignment in fold {fold}")
        splits.append({"fold": fold, "train": train, "validation": validation, "test": test})
    return splits


def run_fold(split, paths, config, device_name, threads):
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    with threadpool_limits(limits=threads):
        epochs = prepare(config["subjects"], paths["raw"], paths["cache"])
        subjects = epochs.trials.subject.to_numpy()
        labels = epochs.trials.label.to_numpy(dtype=np.int64)
        train, valid, test = [np.flatnonzero(np.isin(subjects, split[key]))
                              for key in ("train", "validation", "test")]
        fold = split["fold"]
        destination = paths["out"] / "folds" / f"fold_{fold:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        covariances = {}
        for name in config["models"]:
            model_dir = destination / name
            if (model_dir / "fit.json").is_file():
                print(f"Fold {fold}  {name}  resumed", flush=True)
                continue
            print(f"Fold {fold}  {name}  training", flush=True)
            with TemporaryDirectory(dir=destination) as folder:
                temporary = Path(folder) / name
                temporary.mkdir()
                if name in TRADITIONAL:
                    for band in MODELS[name]:
                        if band not in covariances:
                            covariances[band] = trial_covariances(epochs.x[band])
                    probability, stats, state = train_csp(name, covariances, labels, subjects,
                                                          train, valid, test)
                    np.savez_compressed(temporary / "model.npz", **state)
                else:
                    if covariances:
                        covariances.clear()
                    groups = OCCLUSION if name == "fusion" else None
                    probability, stats, state, history, masks = train_net(
                        name, epochs.x, labels, subjects, train, valid, test, config, config["seed"] + fold,
                        device, groups,
                    )
                    torch.save(state, temporary / "model.pt")
                    pd.DataFrame(history, columns=["learning_rate", "epoch", "train_loss",
                                                  "validation_ba"]).to_csv(temporary / "history.csv", index=False)
                    if groups is not None:
                        pd.DataFrame(masks, index=pd.Index(test, name="trial_id")).to_csv(
                            temporary / "mask_probabilities.csv")
                columns = ["subject", "run", "onset_sec", "label"]
                prediction = epochs.trials.iloc[test][columns].copy()
                prediction["prediction"] = (probability > 0.5).astype(np.int64)
                prediction["p_right"] = probability
                prediction.to_csv(temporary / "predictions.csv")
                stats["fold"] = fold
                stats["model"] = name
                (temporary / "fit.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                temporary.rename(model_dir)
            print(f"Fold {fold}  {name}  done in {stats['train_seconds']:.1f}s", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=PROFILES, default="full")
    ap.add_argument("--subjects", help="Override cohort, e.g. 1-12,30")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--fold", type=int, help="Compute one outer fold; report when all folds exist")
    ap.add_argument("--epochs", type=int, help="Override epoch limit")
    ap.add_argument("--jobs", type=int, default=1, help="Concurrent folds (CPU only)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per training process")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.jobs < 1 or args.threads < 1 or (args.epochs is not None and args.epochs < 1):
        ap.error("jobs, threads and epochs must be positive")
    if args.device == "cuda" and (not torch.cuda.is_available() or args.jobs != 1):
        ap.error("CUDA requires an available PyTorch CUDA device and --jobs 1")
    if args.fold is not None and not 1 <= args.fold <= PROFILES[args.profile]["folds"]:
        ap.error("Requested fold is outside the selected profile")
    if args.prepare_only and args.report_only:
        ap.error("--prepare-only and --report-only are mutually exclusive")
    return args


def main():
    args = parse_args()
    config = dict(PROFILES[args.profile])
    config["subjects"] = parse_subjects(args.subjects or config["subjects"])
    config["models"] = list(dict.fromkeys(args.models))
    config["profile"] = args.profile
    if args.epochs is not None:
        config["epochs"] = args.epochs
    paths = {"raw": args.raw_dir.resolve(), "cache": args.cache_dir.resolve(),
             "out": (args.out or ROOT / "outputs" / args.profile).resolve()}
    epochs = prepare(config["subjects"], paths["raw"], paths["cache"])
    if args.prepare_only:
        print(f"Prepared {len(epochs.trials)} trials in {epochs.cache}")
        return
    out = paths["out"]
    out.mkdir(parents=True, exist_ok=True)
    splits = make_folds(epochs.trials.subject, config["folds"], config["seed"])
    study = {"code_version": "1.0", "config": config, "cache": epochs.cache.name, "splits": splits,
             "device": args.device, "versions": {
                 "python": platform.python_version(), "torch": torch.__version__, "mne": mne.__version__,
                 "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                 "sklearn": sklearn.__version__,
             }}
    manifest = out / "study.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != study:
        raise ValueError("Output directory contains a different experiment setup. Use --out with a new directory.")
    if not manifest.exists():
        manifest.write_text(json.dumps(study, indent=2), encoding="utf-8")
        shutil.copyfile(epochs.cache / "qc.csv", out / "qc.csv")
        shutil.copyfile(epochs.cache / "trials.csv", out / "trials.csv")

    if not args.report_only:
        selected = [split for split in splits if args.fold is None or split["fold"] == args.fold]
        if args.jobs == 1 or len(selected) == 1:
            for split in selected:
                run_fold(split, paths, config, args.device, args.threads)
        else:
            from multiprocessing import get_context

            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(run_fold, split, paths, config, args.device, args.threads)
                           for split in selected]
                for future in futures:
                    future.result()

    missing = [(split["fold"], model) for split in splits for model in config["models"]
               if not (out / "folds" / f"fold_{split['fold']:02d}" / model / "fit.json").is_file()]
    if missing:
        if args.report_only:
            raise ValueError(f"Results are incomplete. Missing fold/model pairs: {missing}")
        print(f"Fold output saved at {out}. {len(missing)} fold/model combinations remain.")
        return

    from eegmi.report import build_report

    build_report(out, epochs, study)
    print(f"Completed report: {out / 'report.md'}")


if __name__ == "__main__":
    # Required by deterministic CUDA matrix operations. CPU runs ignore it.
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
