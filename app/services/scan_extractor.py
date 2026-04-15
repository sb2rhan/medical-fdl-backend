"""
scan_extractor.py
=================
Runtime scan feature extractor for the predict-scan endpoint.

Loads the MedicalResNet3D backbone once (singleton) on the same device as the
main AG-HFD model and exposes a single public function:

    extract_features(file_path, modality, device) -> np.ndarray  # (512,)

Weight resolution order (first existing file wins)
---------------------------------------------------
  1. artifacts/medicalnet_weights.pth          <- canonical production path
  2. weights/resnet_10_23dataset.pth           <- local dev / training repo layout

If neither file exists the function raises FileNotFoundError immediately.
Silent random-weight fallback is intentionally removed — using an uninitialised
backbone in a medical decision support system produces meaningless features and
must never go unnoticed.

Download MedicalNet weights (ResNet-10 23-dataset, ~45 MB):
  https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10
Then place at: artifacts/medicalnet_weights.pth

Supported input formats
-----------------------
  .nii, .nii.gz          NIfTI (MRI / fMRI)
  .dcm                   Single DICOM file — parent folder treated as a series
  <folder with .dcm>     DICOM series folder
  .zip                   Zip archive of a DICOM series folder (auto-extracted)
"""

from __future__ import annotations

import os
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Import preprocessing utilities and architecture from the offline extractor
from app.services.extractor import (
    preprocess,
    load_medicalnet_weights,
    build_resnet10,
    OUTPUT_DIM,
)

# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------

# Canonical paths searched in order — first existing file is used.
_WEIGHTS_CANDIDATES = [
    Path("artifacts") / "medicalnet_weights.pth",   # production
    Path("weights")   / "resnet_10_23dataset.pth",  # local dev / training repo
]


def _resolve_weights_path() -> Path:
    """
    Return the first existing MedicalNet weights file from the candidate list.
    Raises FileNotFoundError if none are found — random init must never be used.
    """
    for p in _WEIGHTS_CANDIDATES:
        if p.exists():
            return p
    searched = ", ".join(str(p) for p in _WEIGHTS_CANDIDATES)
    raise FileNotFoundError(
        "MedicalNet ResNet-10 weights not found. Searched:\n"
        f"  {searched}\n"
        "Download (~45 MB): "
        "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10\n"
        "Then place at: artifacts/medicalnet_weights.pth"
    )


# ---------------------------------------------------------------------------
# Singleton backbone
# ---------------------------------------------------------------------------

_BACKBONE: Optional[torch.nn.Module] = None
_BACKBONE_DEVICE: Optional[torch.device] = None


# Map of common file extensions -> modality guess
_EXT_TO_MODALITY: dict[str, str] = {
    ".nii":  "MRI",
    ".gz":   "MRI",   # .nii.gz
    ".dcm":  "CT",
    ".img":  "MRI",
    ".hdr":  "MRI",
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
    Build MedicalResNet3D and load pretrained weights.
    Raises FileNotFoundError if weights are missing (no silent random init).
    """
    weights_path = _resolve_weights_path()   # raises if not found
    model = build_resnet10(out_dim=OUTPUT_DIM).to(device)
    load_medicalnet_weights(model, str(weights_path))
    model.eval()

    # Warm up CUDA kernels on first load
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
        _BACKBONE = _load_backbone(device)   # raises FileNotFoundError if weights missing
        _BACKBONE_DEVICE = device
    return _BACKBONE


# ---------------------------------------------------------------------------
# DICOM zip unpacking
# ---------------------------------------------------------------------------

def _unzip_dicom(zip_path: str | Path, dest_dir: str | Path) -> str:
    """
    Extract a zip archive of a DICOM series into dest_dir.
    Returns the path of the extracted folder containing .dcm files.
    """
    dest_dir = Path(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    dcm_folders = set()
    for root, _, files in os.walk(dest_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            dcm_folders.add(root)

    if not dcm_folders:
        raise ValueError("Zip archive contains no .dcm files.")

    # Use the shallowest folder (top of the series)
    return sorted(dcm_folders, key=lambda p: p.count(os.sep))[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    file_path: str | Path,
    modality: Optional[str] = None,
    device: Optional[torch.device] = None,
    ct_window_center: float = 40.0,
    ct_window_width:  float = 400.0,
) -> np.ndarray:
    """
    Run the full extraction pipeline on a single scan file.

    Parameters
    ----------
    file_path        : path to .nii, .nii.gz, .dcm, a DICOM folder, or a .zip
    modality         : 'MRI', 'CT', or 'PET'. If None, inferred from extension.
    device           : torch.device to run on. Defaults to CUDA if available.
    ct_window_center : HU window centre (CT only)
    ct_window_width  : HU window width  (CT only)

    Returns
    -------
    features : np.ndarray of shape (512,), L2-normalised float32

    Raises
    ------
    FileNotFoundError  if MedicalNet weights are not found at expected paths
    FileNotFoundError  if scan file does not exist
    ValueError         if the scan cannot be loaded or produces wrong shape
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Scan file not found: {file_path}")

    # Unpack zip archives into a temp directory
    _tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if file_path.suffix.lower() == ".zip":
        _tmp_dir = tempfile.TemporaryDirectory()
        file_path = Path(_unzip_dicom(file_path, _tmp_dir.name))

    try:
        if modality is None:
            modality = _guess_modality(file_path)

        norm_kwargs = (
            {"ct_window_center": ct_window_center,
             "ct_window_width":  ct_window_width}
            if modality.upper() == "CT" else {}
        )

        # load -> normalise -> resample -> (1, 96, 128, 128)
        volume = preprocess(str(file_path), modality=modality, **norm_kwargs)

        # Add batch dim: (1, 1, 96, 128, 128)
        volume = volume.unsqueeze(0).to(device, non_blocking=True)

        backbone = get_backbone(device)  # raises FileNotFoundError if weights missing
        with torch.no_grad():
            with torch.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
                feat = backbone(volume)       # (1, 512)
            feat = feat.float()               # cast back from fp16 if autocast was on

        return feat.squeeze(0).cpu().numpy().astype(np.float32)  # (512,)

    finally:
        if _tmp_dir is not None:
            _tmp_dir.cleanup()
