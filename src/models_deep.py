"""
Stage 3 (main models) — LSTM and a small Transformer encoder, trained on rolling
windows of engineered features to predict the regime of the forecast horizon.

Requires PyTorch. If torch is not installed, `run()` logs a clear message and
returns empty results rather than crashing the pipeline, so `run_all.py` still
produces everything else.

Alignment note: a window ENDING at day t holds the scaled features of days
t-L+1 ... t and is labeled with the target of day t. Windows for the first test
days reach back into the (purged) rows before the test period; that is past
information, so every model - deep or not - is scored on the same test dates.
"""
from __future__ import annotations

import math

import numpy as np

from src.dataset import build_dataset
from src.metrics import classification_scores, pred_frame
from src.utils import load_config, log, resolve_path, set_seed, torch_available


def _require_torch():
    import torch
    import torch.nn as nn

    return torch, nn


class LSTMClassifier:
    def __init__(self, n_features: int, n_classes: int, cfg: dict, seed: int):
        torch, nn = _require_torch()
        self.torch = torch

        class _Net(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, dropout, n_classes):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(hidden_size // 2, n_classes),
                )

            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.head(h_n[-1])  # final layer's final hidden state

        torch.manual_seed(seed)
        self.model = _Net(n_features, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"], n_classes)
        self.cfg = cfg

    def net(self):
        return self.model


class TransformerClassifier:
    def __init__(self, n_features: int, n_classes: int, cfg: dict, seed: int):
        torch, nn = _require_torch()
        self.torch = torch

        class _PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=512):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, : x.size(1)]

        class _Net(nn.Module):
            def __init__(self, n_features, d_model, nhead, num_layers, dim_ff, dropout, n_classes):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                self.pos_enc = _PositionalEncoding(d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                    dropout=dropout, batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes),
                )

            def forward(self, x):
                z = self.encoder(self.pos_enc(self.input_proj(x)))
                return self.head(z.mean(dim=1))  # mean-pool over the sequence

        torch.manual_seed(seed)
        self.model = _Net(n_features, cfg["d_model"], cfg["nhead"], cfg["num_layers"],
                          cfg["dim_feedforward"], cfg["dropout"], n_classes)
        self.cfg = cfg

    def net(self):
        return self.model


def _train_torch_model(wrapper, X_train, y_train, X_val, y_val, cfg: dict, name: str, verbose: bool = True):
    torch = wrapper.torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wrapper.net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    n_classes = int(max(y_train.max(), y_val.max() if len(y_val) else 0)) + 1
    counts = np.bincount(y_train.astype(int), minlength=n_classes)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    weights[counts == 0] = 0.0   # a regime absent from training gets no weight
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    n, batch_size = len(X_train_t), cfg["batch_size"]
    best_val_loss, best_state = float("inf"), None
    patience_left = cfg.get("patience", 8)

    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx].to(device), y_train_t[idx].to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item() if len(X_val_t) else epoch_loss

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.get("patience", 8)
        else:
            patience_left -= 1

        if verbose and (epoch % 5 == 0 or epoch == cfg["epochs"] - 1):
            log(f"  [{name}] epoch {epoch:03d}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}")
        if patience_left <= 0:
            if verbose:
                log(f"  [{name}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def _predict(model, X, device):
    import torch

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(X, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
    return proba


def fit_predict(fold, cfg: dict, verbose: bool = True, save_dir=None) -> dict:
    """Train LSTM + Transformer on one fold; predict that fold's test rows."""
    if not torch_available():
        return {}
    torch, _ = _require_torch()
    seed = cfg.get("random_seed", 42)
    set_seed(seed)

    tr_pos, val_pos = fold.deep_train_val_positions(cfg.get("val_size", 0.1))
    X_tr, y_tr = fold.windows(tr_pos), fold.y_all[tr_pos]
    X_val, y_val = fold.windows(val_pos), fold.y_all[val_pos]
    X_test = fold.windows(fold.test_pos)
    n_features, k = X_tr.shape[2], fold.n_regimes

    out = {}
    for key, wrapper_cls, name in [("lstm", LSTMClassifier, "LSTM"),
                                   ("transformer", TransformerClassifier, "Transformer")]:
        if verbose:
            log(f"Training {name}...")
        wrapper = wrapper_cls(n_features, k, cfg[key], seed)
        model, device = _train_torch_model(wrapper, X_tr, y_tr, X_val, y_val, cfg[key], name, verbose)
        proba = _predict(model, X_test, device)
        out[key] = {"name": name, "preds": proba.argmax(axis=1), "proba": proba,
                    "pred_vol": fold.proba_to_expected_vol(proba)}
        if save_dir is not None:
            torch.save(model.state_dict(), save_dir / f"{key}.pt")
    return out


def run(cfg: dict | None = None):
    cfg = cfg or load_config()
    set_seed(cfg.get("random_seed", 42))

    if not torch_available():
        log("PyTorch not installed - skipping LSTM/Transformer. `pip install torch` to enable.")
        return {}, None

    fold, _ = build_dataset(cfg)
    outputs_dir = resolve_path(cfg, "outputs_dir")
    models_dir = outputs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    results = fit_predict(fold, cfg, verbose=True, save_dir=models_dir)
    for key, res in results.items():
        s = classification_scores(fold.y_test, res["preds"], fold.n_regimes)
        log(f"{res['name']}: accuracy={s['accuracy']:.4f}  macro-F1={s['macro_f1']:.4f}")
        pred_frame(fold, res).to_csv(outputs_dir / f"preds_{key}.csv", index=False)
    return results, fold


if __name__ == "__main__":
    run()
