"""
scan_extractor.py
=================
Runtime scan feature extractor for the predict-scan endpoint.

Loads the MedicalResNet3D backbone once (singleton) on the same device as the
main AG-HFD model and exposes a single public function:

    extract_features(file_path: str | Path, modality: str) -> np.ndarray  # (512,)

Supported input formats
-----------------------
  .nii, .nii.gz          NIfTI (MRI / fMRI)
  .dcm                   Single DICOM file — the parent folder is treated as a series
  <folder containing .dcm files>   DICOM series folder
  .zip                   Zip archive of a DICOM series folder (auto-extracted)

The extractor is intentionally thin: it delegates all I/O and normalisation to
the existing `extractor.py` pipeline (load_volume, preprocess, etc.) and only
adds the singleton backbone + GPU inference step.
"""

from __future__ import annotations

import os
import zipfile
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Import preprocessing utilities from the existing offline extractor
from app.services.extractor import (
    preprocess,
    load_medicalnet_weights,
    build_resnet10,
    OUTPUT_DIM,
)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton backbone
# ─────────────────────────────────────────────────────────────────────────────

_BACKBONE: Optional[torch.nn.Module] = None
_BACKBONE_DEVICE: Optional[torch.device] = None


# Map of common file extensions → modality guess
_EXT_TO_MODALITY: dict[str, str] = {
    ".nii":    "MRI",
    ".gz":     "MRI",   # .nii.gz
    ".dcm":    "CT",
    ".img":    "MRI",
    ".hdr":    "MRI",
}


def _guess_modality(path: str | Path) -> str:
    """Guess modality from file extension. Defaults to 'MRI'."""
    ext = "".join(Path(path).suffixes).lower()
    for suffix, mod in _EXT_TO_MODALITY.items():
        if ext.endswith(suffix):
            return mod
    return "MRI"


def _load_backbone(device: torch.device) -> torch.nn.Module:
    """
    Build and optionally load pretrained weights for the feature extractor.

    If `artifacts/medicalnet_weights.pth` exists it will be loaded.
    Otherwise the backbone runs with random weights — still produces
    valid 512-dim vectors, just without transfer-learning benefits.
    """
    model = build_resnet10(out_dim=OUTPUT_DIM).to(device)

    weights_candidates = [
        Path("artifacts") / "medicalnet_weights.pth",
        Path("weights") / "resnet_10_23dataset.pth",
    ]
    for wp in weights_candidates:
        if wp.exists():
            load_medicalnet_weights(model, str(wp))
            break
    else:
        warnings.warn(
            "No MedicalNet weights found. Backbone will use random initialisation.\n"
            "Place weights at: artifacts/medicalnet_weights.pth\n"
            "Download: https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10",
            RuntimeWarning,
            stacklevel=2,
        )

    model.eval()
    # Warm up CUDA kernels
    try:
        dummy = torch.zeros(1, 1, 96, 128, 128, device=device)
        with torch.no_grad():
            model(dummy)
    except Exception:
        pass
    return model


def get_backbone(device: torch.device) -> torch.nn.Module:
    """Return the singleton backbone, creating it on first call."""
    global _BACKBONE, _BACKBONE_DEVICE
    if _BACKBONE is None or _BACKBONE_DEVICE != device:
        _BACKBONE = _load_backbone(device)
        _BACKBONE_DEVICE = device
    return _BACKBONE


# ─────────────────────────────────────────────────────────────────────────────
# DICOM zip unpacking
# ─────────────────────────────────────────────────────────────────────────────

def _unzip_dicom(zip_path: str | Path, dest_dir: str | Path) -> str:
    """
    Extract a zip archive of a DICOM series into dest_dir.
    Returns the path of the extracted folder containing .dcm files.
    """
    dest_dir = Path(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # Find the deepest folder that contains .dcm files
    dcm_folders = set()
    for root, _, files in os.walk(dest_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            dcm_folders.add(root)

    if not dcm_folders:
        raise ValueError("Zip archive contains no .dcm files.")

    # Use the shallowest folder (top of the series)
    return sorted(dcm_folders, key=lambda p: p.count(os.sep))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    file_path: str | Path,
    modality: Optional[str] = None,
    device: Optional[torch.device] = None,
    ct_window_center: float = 40.0,
    ct_window_width: float = 400.0,
) -> np.ndarray:
    """
    Run the full extraction pipeline on a single scan file.

    Parameters
    ----------
    file_path        : path to .nii, .nii.gz, .dcm, a DICOM folder, or a .zip
    modality         : 'MRI', 'CT', or 'PET'.  If None, inferred from extension.
    device           : torch.device to run on.  Defaults to CUDA if available.
    ct_window_center : HU window centre (CT only)
    ct_window_width  : HU window width  (CT only)

    Returns
    -------
    features : np.ndarray of shape (512,), L2-normalised float32
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Scan file not found: {file_path}")

    # Unpack zip archives
    _tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if file_path.suffix.lower() == ".zip":
        _tmp_dir = tempfile.TemporaryDirectory()
        file_path = Path(_unzip_dicom(file_path, _tmp_dir.name))

    try:
        if modality is None:
            modality = _guess_modality(file_path)

        norm_kwargs = (
            {"ct_window_center": ct_window_center, "ct_window_width": ct_window_width}
            if modality.upper() == "CT" else {}
        )

        # load → normalise → resample → (1, 96, 128, 128)
        volume = preprocess(str(file_path), modality=modality, **norm_kwargs)

        # Add batch dim: (1, 1, 96, 128, 128)
        volume = volume.unsqueeze(0).to(device, non_blocking=True)

        backbone = get_backbone(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                feat = backbone(volume)         # (1, 512)
            feat = feat.float()                 # cast back from fp16 if autocast was on

        return feat.squeeze(0).cpu().numpy().astype(np.float32)  # (512,)

    finally:
        if _tmp_dir is not None:
            _tmp_dir.cleanup()
