"""
predict_scan.py
===============
Multipart /api/predict-scan endpoint.

Clinicians upload raw scan files (.dcm / .nii / .nii.gz / .zip of DICOM
series) directly — no offline pre-extraction step is required.

Prerequisite
------------
MedicalNet ResNet-10 weights (~45 MB) must be placed at:
  artifacts/medicalnet_weights.pth
Download: https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10

Request (multipart/form-data)
-----------------------------
  subject_id   : str               (form field)
  clinical     : str               (JSON form field, ClinicalFeatures)
  MRI          : UploadFile        (optional .nii / .nii.gz)
  CT           : UploadFile        (optional .dcm file, DICOM folder zip, or .nii)
  PET          : UploadFile        (optional .nii / .nii.gz)

Example curl
------------
  curl -X POST http://localhost:8080/api/predict-scan \\
    -H "X-API-Key: $KEY" \\
    -F subject_id=PAT001 \\
    -F 'clinical={"features":{"Age":72,"MMSE":24.0}}' \\
    -F 'MRI=@/data/PAT001_brain.nii.gz'
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.auth import require_api_key
from app.inference import run_single
from app.model_loader import load_artifacts
from app.schemas.predict import ClinicalFeatures, PredictRequest, PredictResponse
from app.services.scan_extractor import extract_features, _guess_modality

router = APIRouter(prefix="/api", tags=["predict"])
_pool  = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# File extensions that are directly accepted
_ACCEPTED_EXTS = {".nii", ".gz", ".dcm", ".img", ".hdr", ".zip"}

# Maximum allowed upload size per scan file (500 MB).
# DICOM series can be several GB; reject oversized uploads early to prevent OOM.
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in _ACCEPTED_EXTS


def _npz_to_b64(features: np.ndarray) -> str:
    """Pack a feature vector into an in-memory .npz and base64-encode it."""
    buf = io.BytesIO()
    np.savez(buf, features=features)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an UploadFile to disk, enforcing the size limit."""
    contents = await upload.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload for '{upload.filename}' exceeds the 500 MB limit "
                f"({len(contents) / 1024 / 1024:.1f} MB received). "
                "Pre-extract NPZ features offline and use /api/predict instead."
            ),
        )
    dest.write_bytes(contents)


@router.post(
    "/predict-scan",
    response_model=PredictResponse,
    summary="Predict from raw scan files (.dcm / .nii / .nii.gz)",
    dependencies=[Depends(require_api_key)],
)
async def predict_from_scan(
    subject_id: str           = Form(..., description="Patient / subject identifier"),
    clinical:   str           = Form(..., description='JSON ClinicalFeatures: {"features": {"Age": 72, ...}}'),
    MRI: Optional[UploadFile] = File(default=None, description="MRI scan (.nii / .nii.gz)"),
    CT:  Optional[UploadFile] = File(default=None, description="CT scan (.dcm, DICOM folder zip, or .nii)"),
    PET: Optional[UploadFile] = File(default=None, description="PET scan (.nii / .nii.gz)"),
):
    # -- 1. Parse and validate clinical JSON ----------------------------------
    try:
        clinical_data = ClinicalFeatures.model_validate(json.loads(clinical))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid clinical JSON: {e}")

    uploaded: dict[str, UploadFile] = {}
    if MRI is not None: uploaded["MRI"] = MRI
    if CT  is not None: uploaded["CT"]  = CT
    if PET is not None: uploaded["PET"] = PET

    if not uploaded:
        raise HTTPException(
            status_code=422,
            detail="At least one scan file must be provided (MRI, CT, or PET field).",
        )

    # -- 2. Validate file extensions ------------------------------------------
    for mod_name, upload in uploaded.items():
        if not _ext_ok(upload.filename or ""):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type for {mod_name}: '{upload.filename}'. "
                    "Accepted: .nii, .nii.gz, .dcm, .zip"
                ),
            )

    arts   = load_artifacts()
    device = arts["device"]

    # -- 3. Early check: fail fast if MedicalNet weights are missing -----------
    # _resolve_weights_path() inside scan_extractor will raise FileNotFoundError
    # during the first extract_features() call. We surface a clean 503 here
    # rather than a raw 500 traceback.
    from app.services.scan_extractor import _resolve_weights_path
    try:
        _resolve_weights_path()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "MedicalNet backbone weights are not installed on this server. "
                "Place resnet_10_23dataset.pth at artifacts/medicalnet_weights.pth. "
                f"Details: {e}"
            ),
        )

    # -- 4. Save uploads to temp dir, extract features, collect b64 -----------
    modalities_b64: dict[str, str] = {}
    tmp_root = tempfile.mkdtemp(prefix="scan_upload_")

    try:
        for mod_name, upload in uploaded.items():
            filename = upload.filename or f"{mod_name}_scan{Path(upload.filename or '').suffix}"
            dest = Path(tmp_root) / mod_name / filename
            dest.parent.mkdir(parents=True, exist_ok=True)

            await _save_upload(upload, dest)

            # Run CPU-bound extraction in the thread pool (keeps event loop free)
            try:
                features: np.ndarray = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        _pool,
                        lambda p=str(dest), m=mod_name, d=device: extract_features(
                            p, modality=m, device=d
                        ),
                    ),
                    timeout=120.0,  # 2 min max per modality
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Feature extraction timed out for modality '{mod_name}'.",
                )
            except FileNotFoundError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not extract features from {mod_name} scan: {e}",
                )

            if features.shape != (512,):
                raise HTTPException(
                    status_code=500,
                    detail=f"Extractor returned unexpected shape {features.shape} for {mod_name}.",
                )

            modalities_b64[mod_name] = _npz_to_b64(features)

        # -- 5. Build a standard PredictRequest and call run_single -----------
        req = PredictRequest(
            subject_id=subject_id,
            clinical=clinical_data,
            modalities=modalities_b64,
        )

        try:
            result: PredictResponse = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    _pool, run_single, req, arts
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Model inference timed out.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return result

    finally:
        # Always clean up temp files regardless of success or failure
        shutil.rmtree(tmp_root, ignore_errors=True)
