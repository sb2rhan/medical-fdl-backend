"""
extractor.py
============
Offline MRI/CT/PET feature extractor using a MedicalNet-pretrained ResNet3D backbone.

Default backbone: ResNet-10 trained on 23 medical datasets
(resnet_10_23dataset.pth — 45 MB, fastest, recommended for CPU).

Produces 512-dim L2-normalized feature vectors saved as .npz files.
OUTPUT_DIM = 512 must match ModalityConfig.input_dim in your AG-HFD checkpoint.

Quick start
-----------
1. Download MedicalNet weights (ResNet-10 23-dataset):
   https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10
   Save as: weights/resnet_10_23dataset.pth

2. Run shape check + full extraction:
   PYTHONPATH=. python app/services/extractor.py

Dependencies
------------
   pip install torch nibabel pydicom
   pip install tqdm          # only needed for offline batch extraction
   pip install SimpleITK     # optional but recommended for robust DICOM loading

NOTE: tqdm is intentionally NOT imported at module level.
It is a training/offline-only dependency. A lazy import inside
extract_and_save_features() ensures the deployment server never
crashes with ImportError just because tqdm is absent from the
production image.
"""

import os
import glob
import json
import datetime
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (must match ModalityConfig.input_dim in your AG-HFD checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIM = 512           # ResNet-10/18/34 backbone output dim — do not change
TARGET_SHAPE = (96, 128, 128)  # (D, H, W) canonical size fed to the CNN

# MedicalNet input normalisation statistics (from the original paper)
MEDICALNET_MEAN = 58.09
MEDICALNET_STD  = 49.73

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — MedicalNet ResNet3D architecture
#
# Layer names match resnet_10_23dataset.pth exactly so that
# load_state_dict() works without key remapping.
# Reference: https://github.com/Tencent/MedicalNet
# ═════════════════════════════════════════════════════════════════════════════


class BasicBlock3D(nn.Module):
    """Standard residual block with two 3×3×3 convolutions."""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2        = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck3D(nn.Module):
    """1×1×1 → 3×3×3 → 1×1×1 bottleneck block (ResNet-50+)."""
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm3d(planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalResNet3D(nn.Module):
    """
    3D ResNet backbone compatible with MedicalNet pretrained weights.

    Supported depths and their layer configs:
      ResNet-10  : layers=[1,1,1,1], block=BasicBlock3D  → 512-d output
      ResNet-18  : layers=[2,2,2,2], block=BasicBlock3D  → 512-d output
      ResNet-34  : layers=[3,4,6,3], block=BasicBlock3D  → 512-d output
      ResNet-50  : layers=[3,4,6,3], block=Bottleneck3D  → 2048-d → projected to 512-d

    The projection head maps backbone output → OUTPUT_DIM (512).
    For ResNet-10/18/34 this is an identity mapping (512→512).
    For ResNet-50 it is a 2048→512 linear layer.
    """

    def __init__(self, block, layers: list[int], out_dim: int = OUTPUT_DIM):
        super().__init__()
        self.inplanes = 64

        # Stem — matches MedicalNet naming exactly
        self.conv1   = nn.Conv3d(1, 64, kernel_size=7,
                                 stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1     = nn.BatchNorm3d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Projection head: backbone_dim → out_dim
        backbone_dim = 512 * block.expansion  # 512 for Basic, 2048 for Bottleneck
        if backbone_dim == out_dim:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Sequential(
                nn.Linear(backbone_dim, out_dim),
                nn.ReLU(inplace=True),
            )

        self._init_weights()

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) normalised volume
        Returns:
            features: (B, out_dim) L2-normalised
        """
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.projector(x)
        return F.normalize(x, p=2, dim=1)  # unit sphere → cosine sim is valid


# ─── Factory functions ────────────────────────────────────────────────────────

def build_resnet10(out_dim: int = OUTPUT_DIM) -> MedicalResNet3D:
    """ResNet-10 23-dataset (resnet_10_23dataset.pth) — default backbone."""
    return MedicalResNet3D(BasicBlock3D, [1, 1, 1, 1], out_dim)

def build_resnet18(out_dim: int = OUTPUT_DIM) -> MedicalResNet3D:
    return MedicalResNet3D(BasicBlock3D, [2, 2, 2, 2], out_dim)

def build_resnet34(out_dim: int = OUTPUT_DIM) -> MedicalResNet3D:
    return MedicalResNet3D(BasicBlock3D, [3, 4, 6, 3], out_dim)

def build_resnet50(out_dim: int = OUTPUT_DIM) -> MedicalResNet3D:
    return MedicalResNet3D(Bottleneck3D, [3, 4, 6, 3], out_dim)


BACKBONE_REGISTRY: dict[str, callable] = {
    "resnet10": build_resnet10,
    "resnet18": build_resnet18,
    "resnet34": build_resnet34,
    "resnet50": build_resnet50,
}


def load_medicalnet_weights(
    model: MedicalResNet3D,
    weights_path: str,
    strict: bool = False,  # False because we added the projector head
) -> MedicalResNet3D:
    """
    Load MedicalNet pretrained weights into the backbone.

    MedicalNet .pth files store weights under a 'state_dict' key with
    a 'module.' prefix from DataParallel training. This function strips
    the prefix before loading so single-GPU / CPU inference works correctly.

    Default weights (resnet_10_23dataset.pth):
      Download: https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10
      Size: ~45 MB — fastest option, recommended for CPU inference.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"MedicalNet weights not found at '{weights_path}'.\n"
            "Download from: https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10"
        )

    ckpt  = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)  # handle both raw and nested formats

    # Strip DataParallel 'module.' prefix
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)

    # Expected missing: projector.* (new head) — warn on anything else
    unexpected_real = [k for k in unexpected if "projector" not in k]
    missing_real    = [k for k in missing    if "projector" not in k]
    if unexpected_real:
        warnings.warn(f"Unexpected keys in checkpoint: {unexpected_real[:5]}")
    if missing_real:
        warnings.warn(f"Missing keys from checkpoint: {missing_real[:5]}")

    n_loaded = len(cleaned) - len(unexpected_real)
    print(f" ✓ Loaded {n_loaded} / {len(cleaned)} pretrained parameters "
          f"({len(missing)} projector keys initialised from scratch)")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Loaders (NIfTI, DICOM, legacy .hdr/.img)
# ═════════════════════════════════════════════════════════════════════════════


def load_nifti(path: str) -> torch.Tensor:
    """
    Load .nii / .nii.gz / .hdr+.img → canonical RAS+ orientation → (1, D, H, W).
    nib.as_closest_canonical() corrects for scanner-specific axis ordering.
    """
    img  = nib.load(path)
    img  = nib.as_closest_canonical(img)            # reorient to RAS+
    data = img.get_fdata(dtype=np.float32).squeeze()

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D volume after squeeze, got {data.shape} ({path})"
        )
    return torch.from_numpy(data).unsqueeze(0)      # (1, D, H, W)


def load_dicom_series(path: str) -> torch.Tensor:
    """
    Load a DICOM series from a folder or a single .dcm file.

    Prefers SimpleITK (more robust multi-frame / enhanced DICOM) and
    falls back to pydicom slice-by-slice stacking when SimpleITK is absent.
    Always applies RescaleSlope / RescaleIntercept → true HU values.
    """
    path   = Path(path)
    folder = path.parent if path.suffix.lower() == ".dcm" else path

    # Try SimpleITK first
    try:
        import SimpleITK as sitk
        reader    = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(str(folder))
        if not dcm_names:
            raise RuntimeError("No DICOM series found")
        reader.SetFileNames(dcm_names)
        itk_image = reader.Execute()
        vol = sitk.GetArrayFromImage(itk_image).astype(np.float32)  # (D, H, W) in HU
        return torch.from_numpy(vol).unsqueeze(0)
    except ImportError:
        pass  # SimpleITK not installed — fall through to pydicom

    # Fallback: pydicom slice stacking
    import pydicom

    dcm_files = sorted(folder.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files in {folder}")

    slices = [pydicom.dcmread(str(f)) for f in dcm_files]

    def _sort_key(ds):
        if hasattr(ds, "InstanceNumber"): return float(ds.InstanceNumber)
        if hasattr(ds, "SliceLocation"):  return float(ds.SliceLocation)
        return 0.0

    slices.sort(key=_sort_key)

    volume = []
    for ds in slices:
        px        = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume.append(px * slope + intercept)  # now in HU

    vol = np.stack(volume, axis=0)              # (D, H, W)
    return torch.from_numpy(vol).unsqueeze(0)   # (1, D, H, W)


def load_volume(path: str) -> tuple[torch.Tensor, str]:
    """
    Dispatch to the right loader based on file extension.
    Returns (tensor (1,D,H,W), detected_format).
    """
    ext = "".join(Path(path).suffixes).lower()

    if ext in (".nii", ".gz", ".nii.gz"):
        return load_nifti(path), "nifti"
    if ext in (".img", ".hdr", ".nifti.hdr", ".nifti.img"):
        return load_nifti(path), "analyze"  # nibabel handles .hdr/.img too
    if ext in (".dcm",):
        return load_dicom_series(path), "dicom"

    raise ValueError(
        f"Unsupported format '{ext}'. Supported: .nii, .nii.gz, "
        f".img/.hdr (Analyze), .dcm (DICOM series folder)"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — Modality-specific normalisation
# ═════════════════════════════════════════════════════════════════════════════


def normalize_mri(tensor: torch.Tensor) -> torch.Tensor:
    """
    Robust percentile-based normalisation.
    Clips to [p1, p99] before scaling to avoid outlier voxels
    (e.g. bright artefacts at the scanner bore edge) dominating the range.
    """
    lo     = torch.quantile(tensor, 0.01)
    hi     = torch.quantile(tensor, 0.99)
    tensor = tensor.clamp(lo, hi)
    # Shift to MedicalNet training stats
    tensor = (tensor - lo) / (hi - lo + 1e-8) * 255.0
    return (tensor - MEDICALNET_MEAN) / (MEDICALNET_STD + 1e-8)


def normalize_ct(
    tensor: torch.Tensor,
    window_center: float = 40.0,
    window_width:  float = 400.0,
) -> torch.Tensor:
    """
    Clinical HU windowing then MedicalNet normalisation.

    Common windows:
      Soft tissue : center=40,   width=400
      Brain       : center=40,   width=80
      Lung        : center=-600, width=1500
      Bone        : center=400,  width=1800
    """
    lo     = window_center - window_width / 2.0
    hi     = window_center + window_width / 2.0
    tensor = tensor.clamp(lo, hi)
    tensor = (tensor - lo) / (hi - lo) * 255.0
    return (tensor - MEDICALNET_MEAN) / (MEDICALNET_STD + 1e-8)


def normalize_pet(tensor: torch.Tensor) -> torch.Tensor:
    """
    PET typically stores SUV values.
    Log-scale compression followed by MedicalNet normalisation.
    """
    tensor = (tensor.clamp(min=0) + 1e-6).log()
    lo     = torch.quantile(tensor, 0.01)
    hi     = torch.quantile(tensor, 0.99)
    tensor = tensor.clamp(lo, hi)
    tensor = (tensor - lo) / (hi - lo + 1e-8) * 255.0
    return (tensor - MEDICALNET_MEAN) / (MEDICALNET_STD + 1e-8)


NORMALIZERS: dict[str, callable] = {
    "MRI": normalize_mri,
    "CT":  normalize_ct,
    "PET": normalize_pet,
}


# ═════════════════════════════════════════════════════════════════════════════
# PART 4 — Spatial resampling
# ═════════════════════════════════════════════════════════════════════════════


def resample_volume(tensor: torch.Tensor,
                   target: tuple = TARGET_SHAPE) -> torch.Tensor:
    """
    Trilinear resample (1, D, H, W) → (1, Dt, Ht, Wt).
    Pure PyTorch — no external registration library needed.
    """
    return F.interpolate(
        tensor.unsqueeze(0),   # (1, 1, D, H, W)
        size=target,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)               # (1, Dt, Ht, Wt)


# ═════════════════════════════════════════════════════════════════════════════
# PART 5 — Unified preprocessing pipeline
# ═════════════════════════════════════════════════════════════════════════════


def preprocess(
    path: str,
    modality: str = "MRI",
    ct_window_center: float = 40.0,
    ct_window_width:  float = 400.0,
) -> torch.Tensor:
    """
    Full pipeline: load → normalise → resample.
    Returns (1, D, H, W) float32 ready for the ResNet3D backbone.
    """
    tensor, _ = load_volume(path)

    norm_fn = NORMALIZERS.get(modality.upper(), normalize_mri)
    if modality.upper() == "CT":
        tensor = norm_fn(tensor, ct_window_center, ct_window_width)
    else:
        tensor = norm_fn(tensor)

    tensor = resample_volume(tensor, TARGET_SHAPE)
    return tensor  # (1, 96, 128, 128)


# ═════════════════════════════════════════════════════════════════════════════
# PART 6 — Dataset discovery helpers  (OFFLINE USE ONLY — not called in deployment)
# ═════════════════════════════════════════════════════════════════════════════


def find_oasis_img_files(disc1_root: str) -> list[dict]:
    """
    Walks the OASIS-1 disc1/ layout and returns records for each subject.

    Expected structure:
      disc1/
        OAS1_XXXX_MR1/
          PROCESSED/MPRAGE/SUBJ_111/
            OAS1_XXXX_MR1_mpr_n4_anon_sbj_111.img
    """
    pattern = os.path.join(
        disc1_root, "OAS1_*", "PROCESSED", "MPRAGE", "SUBJ_111", "*.img"
    )
    paths   = sorted(glob.glob(pattern))
    records = []
    for p in paths:
        normalized = p.replace("\\", "/")
        parts      = normalized.split("/")
        subject_id = next(
            (part for part in parts if part.startswith("OAS1_")), None
        )
        if subject_id is None:
            print(f" ⚠ Could not extract subject ID from: {p} — skipping")
            continue
        records.append({"subject_id": subject_id, "path": p, "modality": "MRI"})
    return records


def find_oasis2_img_files(disc1_root: str) -> list[dict]:
    """
    Finds every MRI visit for every patient in the OASIS-2 dataset.

    Expected layout:
      OASIS2/
        OAS2_0001_MR1/RAW/*.hdr   ← included
        OAS2_0001_MR2/RAW/*.hdr   ← included
    """
    pattern = os.path.join(disc1_root, "OAS2_*", "RAW", "*.hdr")
    paths   = sorted(glob.glob(pattern, recursive=False))
    records = []
    for p in paths:
        normalized = p.replace("\\", "/")
        parts      = normalized.split("/")
        folder     = next(
            (part for part in parts if part.startswith("OAS2_")), None
        )
        if folder is None:
            print(f" ⚠ Could not extract session ID from: {p} — skipping")
            continue
        records.append({"subject_id": folder, "path": p, "modality": "MRI"})
    return records


def find_nifti_files(root: str, modality: str = "MRI") -> list[dict]:
    """
    Generic NIfTI discovery: walk `root` and find all .nii / .nii.gz files.
    Subject ID is inferred from the filename stem.
    """
    records = []
    for ext in ("*.nii", "*.nii.gz"):
        for p in sorted(glob.glob(os.path.join(root, "**", ext), recursive=True)):
            stem = Path(p).stem.replace(".nii", "")  # handle .nii.gz
            records.append({"subject_id": stem, "path": p, "modality": modality})
    return records


def find_dicom_series(root: str, modality: str = "CT") -> list[dict]:
    """
    Find DICOM series by locating folders that contain .dcm files.
    Subject ID = the immediate parent folder name.
    """
    records = []
    seen    = set()
    for p in sorted(glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True)):
        folder = str(Path(p).parent)
        if folder in seen:
            continue
        seen.add(folder)
        records.append({
            "subject_id": Path(folder).name,
            "path":       folder,
            "modality":   modality,
        })
    return records


# ═════════════════════════════════════════════════════════════════════════════
# PART 7 — Batch extraction  (OFFLINE USE ONLY — not called in deployment)
# ═════════════════════════════════════════════════════════════════════════════


def extract_and_save_features(
    records:          list[dict],
    output_dir:       str,
    weights_path:     str,
    backbone:         str   = "resnet10",  # resnet_10_23dataset by default
    device:           Optional[str] = None,
    ct_window_center: float = 40.0,
    ct_window_width:  float = 400.0,
) -> dict:
    """
    Extract features for a list of records and save one .npz per subject.

    Parameters
    ----------
    records          : list of dicts with keys 'subject_id', 'path', 'modality'
    output_dir       : directory to write .npz files and manifest.json
    weights_path     : path to MedicalNet .pth file (resnet_10_23dataset.pth)
    backbone         : one of 'resnet10', 'resnet18', 'resnet34', 'resnet50'
    device           : 'cuda', 'cpu', or None (auto-detect)
    ct_window_center : HU window centre for CT modality
    ct_window_width  : HU window width for CT modality

    Returns
    -------
    manifest : dict  subject_id → {feature_path, status}
    """
    # Lazy import — tqdm is an offline-only dependency not required in deployment.
    # Importing it here (not at module level) means the server will never crash
    # with ImportError if tqdm is absent from the production Docker image.
    from tqdm import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Using device: {dev}")

    # Build and load model
    print(f"Building {backbone} backbone …")
    model = BACKBONE_REGISTRY[backbone](out_dim=OUTPUT_DIM).to(dev)
    print(f"Loading MedicalNet weights from: {weights_path}")
    load_medicalnet_weights(model, weights_path)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    if not records:
        raise ValueError("records list is empty — nothing to extract")

    print(f"\nFound {len(records)} subject(s)")
    manifest, failed = {}, []

    for rec in tqdm(records, desc="Extracting features"):
        sid      = rec["subject_id"]
        path     = rec["path"]
        modality = rec.get("modality", "MRI")
        out_path = os.path.join(output_dir, f"{sid}.npz")

        if os.path.exists(out_path):
            manifest[sid] = {"feature_path": out_path, "status": "skipped (exists)"}
            continue

        try:
            norm_kwargs = (
                {"ct_window_center": ct_window_center,
                 "ct_window_width":  ct_window_width}
                if modality.upper() == "CT" else {}
            )
            tensor = preprocess(path, modality=modality, **norm_kwargs)
            tensor = tensor.unsqueeze(0).to(dev)  # (1, 1, D, H, W)

            with torch.no_grad():
                feat = model(tensor)  # (1, 512)

            feat_np = feat.squeeze(0).cpu().numpy()
            assert feat_np.shape == (OUTPUT_DIM,), \
                f"Unexpected feature shape: {feat_np.shape}"

            np.savez(
                out_path,
                features   = feat_np,
                subject_id = np.array(sid),
                modality   = np.array(modality),
                src_path   = np.array(path),
                timestamp  = np.array(datetime.datetime.now().isoformat()),
            )
            manifest[sid] = {"feature_path": out_path, "status": "ok"}

        except Exception as e:
            print(f"\n ✗ Failed [{sid}]: {e}")
            failed.append({"subject_id": sid, "error": str(e)})
            manifest[sid] = {"feature_path": None, "status": f"error: {e}"}

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    ok   = sum(1 for v in manifest.values() if v["status"] == "ok")
    skip = sum(1 for v in manifest.values() if "skipped" in v["status"])
    print(f"\n{'='*55}")
    print(f" Extraction complete!")
    print(f" ✓ Extracted : {ok}")
    print(f" ⏭ Skipped   : {skip}")
    print(f" ✗ Failed    : {len(failed)}")
    print(f" Manifest    : {manifest_path}")
    print(f"{'='*55}")
    return manifest


# ═════════════════════════════════════════════════════════════════════════════
# PART 8 — Verification utility  (OFFLINE USE ONLY — not called in deployment)
# ═════════════════════════════════════════════════════════════════════════════


def verify_features(output_dir: str) -> None:
    """
    Load all .npz files in output_dir and print per-subject stats.
    Checks: shape, L2 norm (should be ~1.0 since we L2-normalise),
    NaN/Inf presence, and modality label.
    """
    npz_files = sorted(glob.glob(os.path.join(output_dir, "*.npz")))
    if not npz_files:
        print("No .npz files found — nothing to verify.")
        return

    print(f"\nVerifying {len(npz_files)} feature file(s) in '{output_dir}' …\n")
    all_vecs, issues = [], []

    for p in npz_files:
        data     = np.load(p, allow_pickle=True)
        feat     = data["features"]  # explicit key
        sid      = str(data["subject_id"])
        modality = str(data.get("modality", "?"))
        ts       = str(data.get("timestamp", "?"))
        norm     = np.linalg.norm(feat)
        has_nan  = bool(np.isnan(feat).any())
        has_inf  = bool(np.isinf(feat).any())

        status = "OK"
        if has_nan:                     status = "NaN!"
        elif has_inf:                   status = "Inf!"
        elif abs(norm - 1.0) > 0.01:    status = f"norm={norm:.4f} (expected ~1.0)"

        if status != "OK":
            issues.append(f"{sid}: {status}")

        print(f" {sid:35s} shape={feat.shape} norm={norm:.4f} "
              f"modality={modality:4s} {status} [{ts[:19]}]")
        all_vecs.append(feat)

    mat = np.stack(all_vecs)
    print(f"\n{'─'*65}")
    print(f"All features matrix : {mat.shape} (subjects × {OUTPUT_DIM})")
    print(f"Global mean         : {mat.mean():.5f}")
    print(f"Global std          : {mat.std():.5f}")
    print(f"Mean pairwise cosine: {(mat @ mat.T).mean():.4f} (1.0 = identical)")

    if issues:
        print(f"\n⚠ {len(issues)} issue(s) detected:")
        for iss in issues:
            print(f"  • {iss}")
    else:
        print("\n✓ All feature vectors look healthy.")


# ═════════════════════════════════════════════════════════════════════════════
# PART 9 — Local test harness  (OFFLINE USE ONLY — gated by __main__)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── CONFIG — edit these three paths before running ────────────────────
    WEIGHTS_PATH = "weights/resnet_10_23dataset.pth"   # MedicalNet ResNet-10 23-dataset
    DISC1_ROOT   = "/path/to/your/dataset/scans"       # root of your scan dataset
    OUTPUT_DIR   = "/path/to/your/extracted_features"  # output directory for .npz files
    # ─────────────────────────────────────────────────────────────────────

    # 1. Architecture + output shape sanity check
    print("── Shape check ──")
    model_test = build_resnet10(out_dim=OUTPUT_DIM)
    model_test.eval()
    dummy = torch.randn(1, 1, *TARGET_SHAPE)  # (1, 1, 96, 128, 128)
    with torch.no_grad():
        out = model_test(dummy)
    print(f"Input        : {tuple(dummy.shape)}")
    print(f"Output       : {tuple(out.shape)}")         # Expected: (1, 512)
    print(f"L2 norm      : {out.norm(dim=1).item():.4f}")  # Expected: ~1.0
    print(f"OUTPUT_DIM   : {OUTPUT_DIM} (must match ModalityConfig.input_dim)")
    print(f"TARGET_SHAPE : {TARGET_SHAPE} (D, H, W after resampling)")
    print()

    # 2. MedicalNet weights load check
    print("── Weights load check (resnet_10_23dataset) ──")
    if os.path.isfile(WEIGHTS_PATH):
        model_pretrained = build_resnet10(out_dim=OUTPUT_DIM)
        load_medicalnet_weights(model_pretrained, WEIGHTS_PATH)
        print(" ✓ Pretrained weights loaded successfully\n")
    else:
        print(f" ⚠ Weights not found at '{WEIGHTS_PATH}'")
        print(" Download: https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10")
        print(" Continuing with randomly initialised weights for shape check only.\n")

    # 3. Full dataset extraction
    print("── Full dataset extraction ──")
    records = find_oasis2_img_files(DISC1_ROOT)

    if not records:
        print(f" ⚠ No scan files found under '{DISC1_ROOT}'")
        print(" Update DISC1_ROOT in the CONFIG section and check folder structure.")
    else:
        manifest = extract_and_save_features(
            records          = records,
            output_dir       = OUTPUT_DIR,
            weights_path     = WEIGHTS_PATH,
            backbone         = "resnet10",   # swap to resnet18/34/50 for more capacity
            device           = None,          # auto-detect GPU / CPU
        )

    # 4. Verify saved features
    verify_features(OUTPUT_DIR)
