import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from model.architecture import AgnosticHybridFusion
from model.data_config import ModalityConfig

ARTIFACTS = Path("artifacts")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level singleton: lru_cache on a no-arg function would permanently cache
# the CPU result if called before CUDA is initialised.
_ARTIFACTS: dict | None = None

# Canonical MedicalNet weights path exposed so scan_extractor.py can read it
# from the already-loaded artifact dict instead of duplicating path logic.
MEDICALNET_WEIGHTS_PATH = ARTIFACTS / "medicalnet_weights.pth"


def load_artifacts() -> dict:
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    ckpt = torch.load(
        ARTIFACTS / "best_model.pth",
        map_location=DEVICE,
        weights_only=False,
    )

    # -- Rebuild ModalityConfig list from saved dicts --------------------------
    modality_configs = [
        ModalityConfig(
            name=m["name"],
            input_dim=m["input_dim"],
            latent_dim=m["latent_dim"],
        )
        for m in ckpt["modality_configs"]
    ]

    # -- Rebuild model ---------------------------------------------------------
    model = AgnosticHybridFusion(
        modality_configs=modality_configs,
        clinical_dim=len(ckpt["selected_features"]),
        n_fuzzy_sets=ckpt["n_fuzzy_sets"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # -- Rebuild scaler from saved mean/scale ----------------------------------
    scaler = StandardScaler()
    scaler.mean_          = np.array(ckpt["scaler_mean"],  dtype=np.float64)
    scaler.scale_         = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_           = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    # -- Load Platt calibrator saved separately as platt_calibrator.pkl -------
    calibrator = None
    calibrator_path = ARTIFACTS / "platt_calibrator.pkl"
    if calibrator_path.exists():
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)

    # -- GPU warm-up: initialise CUDA kernels before first real request --------
    _warmup(model, modality_configs, len(ckpt["selected_features"]))

    # Cast train_medians values to Python float to avoid numpy dtype mismatches
    # during median imputation in inference.py (_impute_and_scale). Checkpoints
    # may store medians as numpy.float32/float64 depending on the training script.
    raw_medians   = ckpt["train_medians"]
    train_medians = {k: float(v) for k, v in raw_medians.items()}

    _ARTIFACTS = {
        "model":                   model,
        "modality_configs":        modality_configs,
        "scaler":                  scaler,
        "calibrator":              calibrator,
        "features":                ckpt["selected_features"],
        "threshold":               float(ckpt["val_thresh"]),
        "train_medians":           train_medians,
        "n_fuzzy_sets":            ckpt["n_fuzzy_sets"],
        "dataset":                 ckpt["dataset"],
        "target_names":            ckpt["target_names"],
        "device":                  DEVICE,
        "val_bacc":                float(ckpt["val_bacc"]),
        "val_auc":                 float(ckpt["val_auc"]),
        "best_epoch":              int(ckpt["epoch"]),
        # Canonical path for MedicalNet backbone weights used by scan_extractor.
        # scan_extractor.py reads this value so weight path logic lives in one place.
        "medicalnet_weights_path": MEDICALNET_WEIGHTS_PATH,
    }
    return _ARTIFACTS


def _warmup(model, modality_configs, clinical_dim: int) -> None:
    """Single dummy forward pass to initialise CUDA kernels."""
    try:
        dummy = {"clinical": torch.zeros(1, clinical_dim, device=DEVICE)}
        for mcfg in modality_configs:
            dummy[f"mod_{mcfg.name}"]  = torch.zeros(1, mcfg.input_dim, device=DEVICE)
            dummy[f"mask_{mcfg.name}"] = torch.zeros(1, device=DEVICE)
        with torch.no_grad():
            model(dummy)
    except Exception:
        pass  # warmup failure must never block startup
