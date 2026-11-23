"""Read audited EDF+ runs and build a reusable, memory-mapped epoch cache."""
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

from download_data import ROOT, RUNS, expected_files, sha256

CHANNELS = tuple("""
FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6
Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10
TP7 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz
""".split())
BANDS = {"wide": (8, 30), "mu": (8, 13), "beta": (13, 30),
         "beta1": (13, 20), "beta2": (20, 30)}
SUB_BANDS = ("mu", "beta1", "beta2")
LABELS = {"T1": 0, "T2": 1}
RAW_HZ = 160
MODEL_HZ = 80
SECONDS = 4
CACHE_VERSION = 1
TRIAL_COLUMNS = ("subject", "run", "onset_sec", "duration_sec", "event", "label")
QC_COLUMNS = ("subject", "run", "onset_sec", "duration_sec", "event", "status", "reason")
SPECTRAL_BANDS = ("mu", "beta1", "beta2")
FILTERS = {name: butter(4, cutoffs, fs=RAW_HZ, btype="bandpass", output="sos")
           for name, cutoffs in BANDS.items()}


@dataclass
class Epochs:
    x: dict
    trials: pd.DataFrame
    power: np.ndarray
    erd: np.ndarray
    cache: Path


def power_by_band(data):
    frequencies, density = welch(data, fs=RAW_HZ, nperseg=RAW_HZ * 2, axis=-1)
    bands = []
    for name in SPECTRAL_BANDS:
        lo, hi = BANDS[name]
        inside = (frequencies >= lo) & (frequencies < hi)
        bands.append(density[..., inside].sum(axis=-1) * (frequencies[1] - frequencies[0]))
    return np.stack(bands, axis=-2)


def bad_signal(signal):
    if not np.isfinite(signal).all():
        return True
    span = np.ptp(signal, axis=-1)
    return bool((span > 5000).any() or (span < 0.5).sum() > 8)


def read_run(path, subject, run):
    raw = mne.io.read_raw_edf(path, verbose="ERROR")
    if raw.info["sfreq"] != RAW_HZ:
        return [], [(subject, run, None, None, None, "excluded_run",
                     f"unexpected_sfreq_{raw.info['sfreq']:g}_Hz")]
    mne.datasets.eegbci.standardize(raw)
    if len(raw.ch_names) != len(CHANNELS) or set(raw.ch_names) != set(CHANNELS):
        return [], [(subject, run, None, None, None, "excluded_run", "unexpected_channels")]

    signal = raw.get_data(picks=list(CHANNELS)) * 1e6
    annotations = list(zip(raw.annotations.onset, raw.annotations.duration,
                           raw.annotations.description, strict=True))
    accepted = []
    audit = []
    last_end = 0
    if not any(event in LABELS for _, _, event in annotations):
        audit.append((subject, run, None, None, None, "excluded_run", "no_T1_or_T2_annotations"))
    for pos, (onset, duration, event) in enumerate(annotations):
        if event not in LABELS:
            if event != "T0":
                audit.append((subject, run, onset, duration, event, "ignored_event", "unrecognized_event"))
            continue
        start = round((onset - raw.first_time) * RAW_HZ)
        stop = start + SECONDS * RAW_HZ
        next_start = (round((annotations[pos + 1][0] - raw.first_time) * RAW_HZ)
                      if pos + 1 < len(annotations) else signal.shape[-1])
        task = signal[:, start:stop]
        reason = ""
        if round(duration * RAW_HZ) < SECONDS * RAW_HZ:
            reason = "duration_under_4s"
        elif start < 0 or stop > min(next_start, signal.shape[-1]):
            reason = "signal_or_annotation_boundary"
        elif start < last_end:
            reason = "overlapping_trial"
        elif bad_signal(task):
            reason = "nonfinite_flat_or_high_amplitude"
        if reason:
            audit.append((subject, run, onset, duration, event, "excluded_trial", reason))
            continue

        power = power_by_band(task[:, RAW_HZ // 2:RAW_HZ * 5 // 2])
        erd = np.full(power.shape, np.nan, dtype=np.float32)
        rest_start = start - RAW_HZ * 9 // 4
        rest_end = start - RAW_HZ // 4
        if pos > 0 and rest_start >= 0:
            previous, rest_duration, rest_event = annotations[pos - 1]
            previous_start = round((previous - raw.first_time) * RAW_HZ)
            previous_end = previous_start + round(rest_duration * RAW_HZ)
            if rest_event == "T0" and previous_start <= rest_start and rest_end <= previous_end:
                rest = signal[:, rest_start:rest_end]
                if not bad_signal(rest):
                    baseline = power_by_band(rest)
                    erd = (10 * np.log10((power + 1e-12) / (baseline + 1e-12))).astype(np.float32)
        filtered = {
            name: resample_poly(sosfiltfilt(sos, task, axis=-1), MODEL_HZ, RAW_HZ,
                                axis=-1, padtype="line").astype(np.float32)
            for name, sos in FILTERS.items()
        }
        accepted.append(((subject, run, onset, duration, event, LABELS[event]), filtered,
                         power.astype(np.float32), erd))
        last_end = stop
        audit.append((subject, run, onset, duration, event, "accepted", ""))
    return accepted, audit


def write_cache(path, raw_dir, subjects, file_hashes):
    audits, rows, power, erd = [], [], [], []
    datasets = {name: [] for name in BANDS}
    for index, subject in enumerate(subjects, 1):
        subject_trials = []
        for run in RUNS:
            name = f"{subject}/{subject}R{run:02d}.edf"
            file = raw_dir / name
            if not file.is_file() or sha256(file) != file_hashes[name]:
                raise ValueError(f"Missing or corrupt {file}. Run download_data.py or extract the data archive.")
            accepted, audit = read_run(file, subject, run)
            subject_trials.extend(accepted)
            audits.extend(audit)
        if {row[-1] for row, *_ in subject_trials} != set(LABELS.values()):
            audits.append((subject, None, None, None, None, "excluded_subject",
                           "no_usable_trials_or_only_one_class"))
        else:
            for row, features, trial_power, trial_erd in subject_trials:
                rows.append(row)
                power.append(trial_power)
                erd.append(trial_erd)
                for name in BANDS:
                    datasets[name].append(features[name])
        if index % 10 == 0 or index == len(subjects):
            print(f"EDF preparation {index}/{len(subjects)} subjects, {len(rows)} usable trials", flush=True)
    if not rows:
        reasons = pd.DataFrame(audits, columns=QC_COLUMNS)["reason"].value_counts().to_dict()
        raise ValueError(f"No usable subjects in selected cohort. QC reasons: {reasons}")

    table = pd.DataFrame(rows, columns=TRIAL_COLUMNS)
    table.index.name = "trial_id"
    table.to_csv(path / "trials.csv")
    pd.DataFrame(audits, columns=QC_COLUMNS).to_csv(path / "qc.csv", index=False)
    for name, segments in datasets.items():
        np.save(path / f"{name}.npy", np.stack(segments))
    np.save(path / "power.npy", np.stack(power))
    np.save(path / "erd.npy", np.stack(erd))
    (path / "metadata.json").write_text(json.dumps({
        "version": CACHE_VERSION, "sampling_hz": MODEL_HZ, "seconds": SECONDS,
        "channels": CHANNELS, "bands": BANDS, "sha256": file_hashes,
    }, indent=2), encoding="utf-8")


def prepare(subjects, raw_dir, cache_dir):
    from hashlib import sha256 as hash_bytes

    file_hashes = expected_files(subjects)
    fingerprint = hash_bytes(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()[:12]
    cache = cache_dir / f"epochs-v{CACHE_VERSION}-{fingerprint}"
    if not (cache / "metadata.json").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache_dir) as folder:
            temporary = Path(folder) / cache.name
            temporary.mkdir()
            write_cache(temporary, raw_dir, subjects, file_hashes)
            try:
                temporary.rename(cache)
            except OSError:
                if not (cache / "metadata.json").exists():
                    raise
    trials = pd.read_csv(cache / "trials.csv", index_col="trial_id")
    return Epochs({name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in BANDS},
                  trials, np.load(cache / "power.npy"), np.load(cache / "erd.npy"), cache)
