import torch
import numpy as np
from functools import lru_cache
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from model import AgnosticHybridFusion, ModalityConfig

ARTIFACTS = Path("artifacts")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def load_artifacts():
    ckpt = torch.load(
        ARTIFACTS / "best_model.pth",
        map_location=DEVICE,
        weights_only=False,       # needed since checkpoint has non-tensor objects
    )

    # ── Rebuild ModalityConfig list from saved dicts ──────────────────────────
    modality_configs = [
        ModalityConfig(
            name      = m["name"],
            input_dim = m["input_dim"],
            latent_dim= m["latent_dim"],
        )
        for m in ckpt["modality_configs"]
    ]

    # ── Rebuild model ─────────────────────────────────────────────────────────
    model = AgnosticHybridFusion(
        modality_configs = modality_configs,
        clinical_dim     = len(ckpt["selected_features"]),
        n_fuzzy_sets     = ckpt["n_fuzzy_sets"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Rebuild scaler from saved mean/scale ──────────────────────────────────
    scaler = StandardScaler()
    scaler.mean_  = np.array(ckpt["scaler_mean"],  dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_   = scaler.scale_ ** 2              # needed by some sklearn internals
    scaler.n_features_in_ = len(scaler.mean_)

    return {
        "model"           : model,
        "modality_configs": modality_configs,
        "scaler"          : scaler,
        "features"        : ckpt["selected_features"],   # list[str]
        "threshold"       : float(ckpt["val_thresh"]),
        "train_medians"   : ckpt["train_medians"],       # dict[str, float]
        "n_fuzzy_sets"    : ckpt["n_fuzzy_sets"],
        "dataset"         : ckpt["dataset"],
        "target_names"    : ckpt["target_names"],
        "device"          : DEVICE,
        # expose training metrics for /metadata
        "val_bacc"        : float(ckpt["val_bacc"]),
        "val_auc"         : float(ckpt["val_auc"]),
        "best_epoch"      : int(ckpt["epoch"]),
    }