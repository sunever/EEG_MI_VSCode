"""CSP/LDA and EEGNet-style classifiers for the same preprocessed EEG trials."""
from copy import deepcopy
import time

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
from torch import nn
from torch.nn import functional as F

from eegmi.data import CHANNELS, MODEL_HZ, SECONDS, SUB_BANDS

MODELS = {
    "csp_lda": ("wide",),
    "fbcsp_lda": SUB_BANDS,
    "wide": ("wide",),
    "mu": ("mu",),
    "beta": ("beta",),
    "fusion": SUB_BANDS,
    "wide_matched": ("wide",),
    "weighted": SUB_BANDS,
}
TRADITIONAL = ("csp_lda", "fbcsp_lda")
BASE_WIDTH = 4
DEPTH = 2


def subjects_ba(labels, predictions, subjects):
    _, subject_index = np.unique(subjects, return_inverse=True)
    cell = 2 * subject_index + labels
    denominator = np.bincount(cell)
    if (denominator == 0).any():
        raise ValueError("All evaluated subjects must have both classes.")
    return (np.bincount(cell, weights=predictions == labels) / denominator).mean()


def trial_covariances(signal, block=128):
    covariance = np.empty((len(signal), signal.shape[1], signal.shape[1]), dtype=np.float32)
    for start in range(0, len(signal), block):
        stop = start + block
        centered = signal[start:stop] - signal[start:stop].mean(axis=-1, keepdims=True)
        covariance[start:stop] = centered @ centered.transpose(0, 2, 1) / centered.shape[-1]
    return covariance


def csp_filters(covariance, y):
    trace = np.trace(covariance, axis1=1, axis2=2)
    normalized = covariance / np.maximum(trace, 1e-12)[:, None, None]
    n = covariance.shape[1]
    within_class = [0.9 * normalized[y == label].mean(axis=0, dtype=np.float64)
                    + 0.1 * np.eye(n) / n for label in (0, 1)]
    _, vectors = eigh(within_class[0], within_class[0] + within_class[1])
    order = np.ravel(np.column_stack([np.arange(n // 2), np.arange(n - 1, n // 2 - 1, -1)]))
    return vectors[:, order].T


def csp_features(covariance, filters):
    energy = np.einsum("qc,ncd,qd->nq", filters, covariance, filters, optimize=True)
    log_power = np.log(np.maximum(energy, 1e-12))
    return log_power


def train_csp(name, covariances, labels, subjects, train, valid, test):
    started = time.perf_counter()
    banks = MODELS[name]
    filters = [csp_filters(covariances[band][train], labels[train])[:8] for band in banks]
    best = None
    for components in (4, 6, 8):
        train_features = np.concatenate([
            csp_features(covariances[band][train], spatial[:components])
            for band, spatial in zip(banks, filters, strict=True)
        ], axis=1)
        valid_features = np.concatenate([
            csp_features(covariances[band][valid], spatial[:components])
            for band, spatial in zip(banks, filters, strict=True)
        ], axis=1)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(train_features, labels[train])
        score = subjects_ba(labels[valid], lda.predict(valid_features), subjects[valid])
        if best is None or score > best[0]:
            best = score, components, lda
    score, components, lda = best
    train_seconds = time.perf_counter() - started
    selected = np.stack([spatial[:components] for spatial in filters])
    test_features = np.concatenate([
        csp_features(covariances[band][test], spatial)
        for band, spatial in zip(banks, selected, strict=True)
    ], axis=1)
    probability = expit(lda.decision_function(test_features))
    return probability, {
        "train_seconds": train_seconds, "validation_ba": score, "components_per_band": components,
        "best_epoch": None, "learning_rate": None, "trainable_parameters": None,
        "stored_model_coefficients": int(selected.size + lda.coef_.size + lda.intercept_.size),
    }, {"filters": selected, "lda_coef": lda.coef_, "lda_intercept": lda.intercept_,
        "bands": np.asarray(banks)}


class EEGEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        full = width * DEPTH
        self.width = width
        self.temporal = nn.Conv1d(1, width, 33, padding=16, bias=False)
        self.spatial = nn.Linear(len(CHANNELS), full, bias=False)
        self.tail = nn.Sequential(
            nn.BatchNorm2d(full), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.35),
            nn.Conv2d(full, full, (1, 9), padding=(0, 4), groups=full, bias=False),
            nn.Conv2d(full, full, (1, 1), bias=False),
            nn.BatchNorm2d(full), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.35),
            nn.Flatten(),
        )

    def encode_projected(self, projected):
        # Each pair of spatial projections shares a learned temporal filter.
        kernels = self.temporal.weight.repeat_interleave(DEPTH, dim=0)
        filtered = F.conv1d(projected, kernels, padding=16, groups=self.width * DEPTH)
        return self.tail(filtered.unsqueeze(2))

    def forward(self, x):
        return self.encode_projected(self.spatial(x.transpose(1, 2)).transpose(1, 2))

    def masked(self, x, groups):
        """Use linear spatial projections to share work across channel masks."""
        electrodes = x.transpose(1, 2)
        baseline = self.spatial(electrodes)
        batch = x.size(0)
        for start in range(0, len(groups), 12):
            chunk = []
            for group in groups[start:start + 12]:
                correction = F.linear(electrodes[:, :, group], self.spatial.weight[:, group])
                chunk.append(baseline - correction)
            projected = torch.cat(chunk).transpose(1, 2)
            yield self.encode_projected(projected).reshape(len(chunk), batch, -1)


class EEGNet(nn.Module):
    def __init__(self, n_bands, width=BASE_WIDTH, weighted=False):
        super().__init__()
        self.encoders = nn.ModuleList([EEGEncoder(width) for _ in range(n_bands)])
        branch_dim = width * DEPTH * (MODEL_HZ * SECONDS // 32)
        self.gate = nn.Linear(n_bands * branch_dim, n_bands) if weighted else None
        self.head = nn.Linear(branch_dim if weighted else n_bands * branch_dim, 2)

    def classify(self, features):
        joined = torch.cat(features, dim=-1)
        if self.gate is None:
            return self.head(joined)
        weights = torch.softmax(self.gate(joined), dim=-1)
        return self.head((torch.stack(features, dim=-2) * weights.unsqueeze(-1)).sum(dim=-2))

    def forward(self, inputs):
        return self.classify([encoder(x) for encoder, x in zip(self.encoders, inputs, strict=True)])

    def masked(self, inputs, groups):
        branch_outputs = [encoder.masked(x, groups)
                          for encoder, x in zip(self.encoders, inputs, strict=True)]
        for group_features in zip(*branch_outputs, strict=True):
            yield self.classify(group_features)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def matched_width():
    target = count_parameters(EEGNet(3))
    return min(range(BASE_WIDTH, 25),
               key=lambda width: abs(count_parameters(EEGNet(1, width)) - target))


def fit_scaler(data, indices, bands):
    stats = {}
    for band in bands:
        total = np.zeros(len(CHANNELS), dtype=np.float64)
        squared = np.zeros_like(total)
        for start in range(0, len(indices), 128):
            block = data[band][indices[start:start + 128]]
            total += block.sum(axis=(0, 2), dtype=np.float64)
            squared += np.square(block).sum(axis=(0, 2), dtype=np.float64)
        samples = len(indices) * data[band].shape[-1]
        mean = total / samples
        scale = np.sqrt(np.maximum(squared / samples - mean ** 2, 1e-8))
        stats[band] = mean.astype(np.float32)[None, :, None], scale.astype(np.float32)[None, :, None]
    return stats


def batch_inputs(data, indices, scaler, device, bands):
    return tuple(
        torch.from_numpy((data[band][indices] - scaler[band][0]) / scaler[band][1]).to(device)
        for band in bands
    )


@torch.inference_mode()
def predict(net, data, indices, scaler, device, bands, batch_size, occlusion=None):
    net.eval()
    probabilities = []
    conditions = {}
    for start in range(0, len(indices), batch_size):
        inputs = batch_inputs(data, indices[start:start + batch_size], scaler, device, bands)
        probabilities.append(net(inputs).softmax(-1)[:, 1].cpu().numpy())
        if occlusion is not None:
            for i, band in enumerate(bands):
                masked = list(inputs)
                masked[i] = torch.zeros_like(masked[i])
                name = f"drop_{band}"
                conditions.setdefault(name, []).append(net(masked).softmax(-1)[:, 1].cpu().numpy())
            names, groups = zip(*occlusion.items(), strict=True)
            masked = list(net.masked(inputs, groups))
            all_probs = torch.cat(masked).softmax(-1)[..., 1].cpu().numpy()
            for name, probability in zip(names, all_probs, strict=True):
                conditions.setdefault(name, []).append(probability)
    return np.concatenate(probabilities), {name: np.concatenate(values)
                                            for name, values in conditions.items()}


def train_net(name, data, labels, subjects, train, valid, test, config, seed, device, occlusion):
    started = time.perf_counter()
    bands = MODELS[name]
    scaler = fit_scaler(data, train, bands)
    width = matched_width() if name == "wide_matched" else BASE_WIDTH
    best = None
    history = []
    for lr in config["learning_rates"]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = EEGNet(len(bands), width=width, weighted=name == "weighted").to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        best_candidate = -np.inf
        stale = 0
        for epoch in range(1, config["epochs"] + 1):
            net.train()
            shuffled = rng.permutation(train)
            loss_sum = 0.
            for start in range(0, len(shuffled), config["batch_size"]):
                batch = shuffled[start:start + config["batch_size"]]
                inputs = batch_inputs(data, batch, scaler, device, bands)
                target = torch.from_numpy(labels[batch]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(net(inputs), target)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(batch)
            probabilities, _ = predict(net, data, valid, scaler, device, bands, config["batch_size"])
            score = subjects_ba(labels[valid], probabilities > 0.5, subjects[valid])
            history.append((lr, epoch, loss_sum / len(train), score))
            if best is None or score > best[0]:
                best = score, lr, epoch, deepcopy(net.state_dict())
            if score > best_candidate:
                best_candidate, stale = score, 0
            else:
                stale += 1
                if stale >= config["patience"]:
                    break
    validation_ba, lr, epoch, state = best
    train_seconds = time.perf_counter() - started
    net.load_state_dict(state)
    probability, masked = predict(net, data, test, scaler, device, bands,
                                  config["batch_size"], occlusion)
    stats = {"train_seconds": train_seconds, "validation_ba": validation_ba,
             "trainable_parameters": count_parameters(net), "stored_model_coefficients": None,
             "best_epoch": epoch, "learning_rate": lr, "components_per_band": None}
    checkpoint = {"state_dict": {key: value.cpu() for key, value in state.items()},
                  "model": name, "width": width, "bands": bands,
                  "channels": CHANNELS, "sampling_hz": MODEL_HZ, "seconds": SECONDS,
                  "scaler": {band: [torch.from_numpy(value) for value in scaler[band]] for band in bands}}
    return probability, stats, checkpoint, history, masked
