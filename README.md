# Medical FDL Backend

> **FastAPI backend for the Domain-Agnostic Fuzzy Deep Learning (DFDL) Framework** — a multimodal clinical decision support system that fuses interpretable ANFIS fuzzy logic with a deep 3D image encoder, exposing predictions, fuzzy rule explanations, and a RAG-powered clinical copilot via a REST API.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Local Setup (Python)](#local-setup-python)
- [Local Setup (Docker)](#local-setup-docker)
- [Environment Variables](#environment-variables)
- [Model Artifacts](#model-artifacts)
- [API Reference](#api-reference)
- [Authentication](#authentication)
- [Frontend Integration](#frontend-integration)
- [Project Notes](#project-notes)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Application                  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  API Layer   │  │ Service Layer│  │ Model Adapter│  │
│  │              │  │              │  │    Layer     │  │
│  │ predict.py   │  │ prediction   │  │              │  │
│  │ predict_scan │→ │  _service.py │→ │ model_loader │  │
│  │ explain.py   │  │ copilot      │  │ inference.py │  │
│  │ copilot.py   │  │  _service.py │  │              │  │
│  │ health.py    │  │ chroma_store │  └──────────────┘  │
│  │ metadata.py  │  │ llm_client   │                    │
│  └──────────────┘  └──────────────┘                    │
│                                                         │
│         ┌─────────────┐    ┌──────────────────┐        │
│         │  ChromaDB   │    │  NVIDIA NIM      │        │
│         │ Vector Store│    │  Llama 3.1 (RAG) │        │
│         └─────────────┘    └──────────────────┘        │
└─────────────────────────────────────────────────────────┘
         ↑
  artifacts/
  best_model.pth        ← AgnosticHybridFusion checkpoint
  platt_calibrator.pkl  ← Platt probability calibrator
```

The model itself is a **two-stream hybrid**:
- **ANFIS stream** — TSK Adaptive Neuro-Fuzzy Inference System on L1-selected clinical tabular features. Produces intrinsically interpretable IF-THEN fuzzy rules.
- **FlatEncoder stream** — Residual MLP on 512-dim L2-normalized image feature vectors (pre-extracted offline by a frozen MedicalNet 3D-ResNet10 backbone).
- **Fusion Gate** — 2-layer MLP that dynamically weights both streams per sample. Falls back to clinical-only when no imaging is provided.
- **Platt Calibrator** — Logistic regression layer fit post-training to align predicted probabilities with empirical frequencies.

---

## Directory Structure

```
medical-fdl-backend/
├── app/
│   ├── main.py                  # FastAPI app entry point, lifespan, CORS, router registration
│   ├── inference.py             # run_single(): preprocessing → forward pass → response assembly
│   ├── model_loader.py          # Singleton checkpoint loader; rebuilds model, scaler, calibrator
│   ├── api/
│   │   ├── predict.py           # POST /api/predict  (NPZ features, JSON clinical)
│   │   ├── predict_scan.py      # POST /api/predict-scan  (raw .nii / .dcm upload)
│   │   ├── explain.py           # POST /api/rules  (rules + full MF parameters)
│   │   ├── copilot.py           # POST /api/copilot  (RAG Q&A)
│   │   ├── health.py            # GET  /health  (model, CUDA, ChromaDB, LLM key)
│   │   ├── metadata.py          # GET  /api  (dataset/model metadata)
│   │   ├── llm_debug.py         # ⚠ DEBUG ONLY — not registered in main.py
│   │   └── rag_debug.py         # ⚠ DEBUG ONLY — not registered in main.py
│   ├── services/
│   │   ├── copilot_service.py   # Abstention logic, retrieval query builder, LLM prompt
│   │   ├── chroma_store.py      # ChromaDB wrapper: seed, query
│   │   ├── llm_client.py        # NVIDIA NIM async chat client
│   │   ├── scan_extractor.py    # On-the-fly MedicalNet feature extraction from raw scans
│   │   ├── document_loader.py   # Loads medical literature into ChromaDB
│   │   ├── prediction_service.py# Thin orchestration shim
│   │   ├── model_adapter.py     # Model interface adapter
│   │   └── extractor.py        # Import shim for scan_extractor
│   ├── schemas/
│   │   ├── predict.py           # PredictRequest/Response, FuzzyRule, FusionWeights, Metadata
│   │   └── copilot.py           # CopilotRequest/Response
│   └── core/
│       ├── config.py            # Pydantic settings (reads .env)
│       └── auth.py              # API key dependency (X-API-Key header)
├── model/
│   ├── architecture.py          # AgnosticHybridFusion, ANFIS, FlatEncoder, FusionGate
│   └── data_config.py           # ModalityConfig, DatasetConfig definitions
├── artifacts/
│   ├── oasis1/
│   │   ├── best_oasis_model.pt
│   │   └── best_oasis_model_calibrator.pkl
│   └── oasis2/
│       ├── best_oasis_model.pt
│       └── best_oasis_model_calibrator.pkl
│   # ↑ Copy one of these to artifacts/best_model.pth + platt_calibrator.pkl at root
├── chroma_data/                 # ChromaDB persistence directory
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | 3.10 may work but is untested |
| pip | 23+ recommended |
| Git | For cloning |
| NVIDIA NIM API Key | Free tier at [build.nvidia.com](https://build.nvidia.com) |
| NVIDIA GPU + CUDA 12.1 | Optional — CPU inference works but is slower |
| Docker | Only for Option B |

---

## Local Setup (Python)

### 1. Clone the repository

```bash
git clone https://github.com/sb2rhan/medical-fdl-backend.git
cd medical-fdl-backend
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values (see [Environment Variables](#environment-variables) below).

### 5. Place model artifacts

`model_loader.py` expects the checkpoint at `artifacts/best_model.pth` and the Platt calibrator at `artifacts/platt_calibrator.pkl` at the repo root. The repo ships models under dataset-named subfolders. Copy the one you want to serve:

```bash
# Use OASIS-1 (Alzheimer's cross-sectional)
cp artifacts/oasis1/best_oasis_model.pt      artifacts/best_model.pth
cp artifacts/oasis1/best_oasis_model_calibrator.pkl  artifacts/platt_calibrator.pkl

# Or use OASIS-2 (Alzheimer's longitudinal)
cp artifacts/oasis2/best_oasis_model.pt      artifacts/best_model.pth
cp artifacts/oasis2/best_oasis_model_calibrator.pkl  artifacts/platt_calibrator.pkl
```

### 6. Start the server

```bash
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

> `PYTHONPATH=.` is required because `model/architecture.py` is imported as a top-level package (`from model.architecture import ...`).
> `--reload` enables hot-reload during development; remove it in production.

The server will be available at **`http://localhost:8080`**.
Swagger UI (interactive API docs): **`http://localhost:8080/docs`**
ReDoc: **`http://localhost:8080/redoc`**

---

## Local Setup (Docker)

The Dockerfile uses `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` as base. On CPU-only machines PyTorch automatically falls back to CPU.

```bash
# Build the image
docker build -t medical-fdl-backend .

# Run (CPU)
docker run --rm -p 8080:8080 --env-file .env medical-fdl-backend

# Run (GPU — requires nvidia-container-toolkit)
docker run --rm --gpus all -p 8080:8080 --env-file .env medical-fdl-backend
```

> The Dockerfile starts uvicorn with `--workers 1` to prevent VRAM exhaustion from multiple GPU processes.

---

## Environment Variables

All variables are loaded from `.env` via `pydantic-settings`. The app raises a `ValidationError` at startup if any **required** variable is missing.

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | ✅ | — | API key from [build.nvidia.com](https://build.nvidia.com) for NVIDIA NIM LLM inference |
| `API_KEY` | ✅ | — | Shared secret for client authentication. Clients must send `X-API-Key: <value>` |
| `NVIDIA_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | NVIDIA NIM base URL |
| `NVIDIA_MODEL` | No | `meta/llama-3.1-70b-instruct` | LLM model identifier |
| `ALLOWED_ORIGINS` | No | *(empty)* | Comma-separated list of allowed CORS origins, e.g. `http://localhost:3000` |
| `APP_ENV` | No | `production` | Environment label (`development` / `production`) |

**Example `.env`:**
```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
API_KEY=my-local-dev-secret
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
APP_ENV=development
```

---

## Model Artifacts

The model checkpoint is a single `.pth` file saved by the training pipeline. It bundles all inference-time state into one object:

| Key in checkpoint | Description |
|---|---|
| `model_state_dict` | `AgnosticHybridFusion` weights |
| `modality_configs` | List of `{name, input_dim, latent_dim}` dicts |
| `selected_features` | List of L1-selected clinical feature names |
| `n_fuzzy_sets` | Number of fuzzy sets per feature (2 or 3) |
| `scaler_mean` / `scaler_scale` | `StandardScaler` parameters for clinical features |
| `train_medians` | Per-feature training medians for missing value imputation |
| `val_thresh` | Youden's J optimal classification threshold |
| `val_bacc` / `val_auc` | Validation metrics from training |
| `epoch` | Best epoch index |
| `dataset` / `target_names` | Dataset label and class names |

The Platt calibrator (`.pkl`) is a `sklearn.linear_model.LogisticRegression` instance fit on validation logits to align probabilities with empirical frequencies.

---

## API Reference

### Authentication

All `/api/*` endpoints require an `X-API-Key` header matching the `API_KEY` env variable. The `/health` endpoint is public.

```
X-API-Key: your-api-key
```

---

### `GET /health`

Public health check. Returns status of model, CUDA, ChromaDB, and LLM key.

**Response:**
```json
{
  "status": "ok",
  "checks": {
    "model": "ok",
    "cuda": "cpu_only",
    "chromadb": "ok",
    "llm_key": "ok"
  }
}
```

`status` is `"ok"` when all checks pass, `"degraded"` otherwise.

---

### `GET /api`

Returns metadata about the loaded model and dataset.

**Response:**
```json
{
  "dataset_name": "OASIS-1",
  "modalities": ["MRI"],
  "clinical_features": ["MMSE", "nWBV", "Age", "Educ"],
  "n_fuzzy_sets": 2,
  "threshold": 0.4821,
  "target_names": ["NonDemented", "Demented"],
  "device": "cpu",
  "best_epoch": 47,
  "val_bacc": 0.9052,
  "val_auc": 0.9593,
  "calibrator_loaded": true
}
```

---

### `POST /api/predict`

Run inference on pre-extracted NPZ image features + clinical tabular data.

**Request body:**
```json
{
  "subject_id": "OAS1_0001",
  "clinical": {
    "features": {
      "MMSE": 24.0,
      "nWBV": 0.72,
      "Age": 75.0
    }
  },
  "modalities": {
    "MRI": "<base64-encoded .npz bytes>"
  }
}
```

- `modalities` is optional. Omit or leave empty `{}` for clinical-only inference (fusion gate automatically sets visual weight to 0).
- Each modality value is a **base64-encoded `.npz` file** containing a `(512,)` float32 feature vector (output of the offline MedicalNet extractor).

**Response:**
```json
{
  "subject_id": "OAS1_0001",
  "prediction": 1,
  "probability": 0.8341,
  "threshold": 0.4821,
  "fusion_weights": {
    "w_clinical": 0.9123,
    "w_visual": 0.0877
  },
  "anfis_rules": [
    {"conditions": "MMSE=LOW & nWBV=LOW", "strength": 0.7214},
    {"conditions": "MMSE=LOW & nWBV=MED", "strength": 0.1823},
    {"conditions": "MMSE=MED & nWBV=LOW", "strength": 0.0631}
  ],
  "modality_status": {
    "MRI": "present"
  }
}
```

---

### `POST /api/predict-batch`

Batch version of `/api/predict`. Maximum **64 samples** per request.

**Request body:**
```json
{
  "samples": [
    { "subject_id": "S1", "clinical": {"features": {"MMSE": 24}}, "modalities": {} },
    { "subject_id": "S2", "clinical": {"features": {"MMSE": 18}}, "modalities": {} }
  ]
}
```

**Response:** `{ "results": [ <PredictResponse>, ... ] }`

---

### `POST /api/predict-scan`

Upload raw scan files directly (no offline NPZ extraction needed). Accepts `multipart/form-data`.

**Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `subject_id` | form string | ✅ | Patient identifier |
| `clinical` | form string (JSON) | ✅ | `{"features": {"age": 72, ...}}` |
| `MRI` | file | one of these | `.nii` / `.nii.gz` |
| `CT` | file | one of these | `.dcm`, `.zip` of DICOM series, or `.nii` |
| `PET` | file | one of these | `.nii` / `.nii.gz` |

> **Size limit:** 500 MB per file. Larger uploads are rejected with HTTP 413. For larger scans, pre-extract features offline and use `/api/predict` instead.

**Example (curl):**
```bash
curl -X POST http://localhost:8080/api/predict-scan \
  -H "X-API-Key: $API_KEY" \
  -F subject_id=PAT001 \
  -F 'clinical={"features":{"Age":72,"MMSE":24.0}}' \
  -F 'MRI=@/data/PAT001_brain.nii.gz'
```

Returns the same `PredictResponse` shape as `/api/predict`.

---

### `POST /api/rules`

Returns fired fuzzy rules **plus** full ANFIS membership function parameters for every feature and fuzzy set. Useful for deep interpretability inspection.

**Request:** Same as `PredictRequest` (see `/api/predict`).

**Response includes (additional to predict):**
```json
{
  "clinical_features_used": ["MMSE", "nWBV", "Age"],
  "membership_functions": [
    {
      "feature": "MMSE",
      "sets": [
        {
          "set_index": 0,
          "label": "LOW",
          "gaussian": {"mu": 18.4, "sigma": 3.2},
          "trapezoid": {"a": 10.0, "b": 14.0, "c": 20.0, "d": 24.0},
          "blend_weight": 0.6712
        }
      ]
    }
  ]
}
```

---

### `POST /api/copilot`

RAG-powered clinical Q&A. Retrieves relevant medical literature from ChromaDB and generates a grounded natural-language explanation via NVIDIA NIM Llama 3.1. Includes an **abstention mechanism** — if retrieved documents don't align with the model's ANFIS rules, the system abstains instead of hallucinating.

**Request body:**
```json
{
  "question": "Why was this patient classified as high dementia risk?",
  "explanation_payload": {
    "subject_id": "OAS1_0001",
    "prediction": 1,
    "probability": 0.8341,
    "threshold": 0.4821,
    "fusion_weights": {"w_clinical": 0.91, "w_visual": 0.09},
    "anfis_rules": [
      {"conditions": "MMSE=LOW & nWBV=LOW", "strength": 0.72}
    ],
    "modality_status": {"MRI": "present"}
  }
}
```

> `explanation_payload` must be a `PredictResponse` object. The `anfis_rules` field is critical — the abstention mechanism uses rule condition terms to validate retrieved context.

**Response:**
```json
{
  "summary": "The patient shows strong markers of dementia based on low MMSE and reduced brain volume.",
  "model_rationale": "ANFIS rule MMSE=LOW & nWBV=LOW fired with strength 0.72, indicating high dementia risk.",
  "evidence": [
    "MMSE scores below 24 are associated with mild cognitive impairment.",
    "Reduced normalized whole brain volume (nWBV) is a structural biomarker for neurodegeneration."
  ],
  "citations": ["rule_mmse_low", "feature_nwbv"],
  "limitations": "Findings are based on cross-sectional data; longitudinal follow-up is recommended.",
  "uncertainty": "Moderate. Retrieved context partially covers the query."
}
```

---

## Authentication

All `/api/*` endpoints use a simple shared API key scheme. Pass the key in the request header:

```
X-API-Key: your-api-key-from-.env
```

The key is validated against the `API_KEY` environment variable by the `require_api_key` FastAPI dependency in `app/core/auth.py`. Requests without a valid key receive HTTP `403 Forbidden`.

---

## Frontend Integration

The frontend repo is at [sb2rhan/medical-fdl-tool](https://github.com/sb2rhan/medical-fdl-tool).

**Integration checklist:**

1. **CORS** — Add the frontend's origin to `ALLOWED_ORIGINS` in `.env`:
   ```env
   ALLOWED_ORIGINS=http://localhost:3000,https://your-deployed-frontend.com
   ```

2. **Auth header** — All fetch/axios calls to `/api/*` must include:
   ```js
   headers: { 'X-API-Key': import.meta.env.VITE_API_KEY }
   ```

3. **Copilot payload shape** — The `explanation_payload` sent to `/api/copilot` must use the `anfis_rules` field (list of `{conditions, strength}` objects) from the `PredictResponse`. Do **not** use legacy field names like `fired_rules` or `top_features` — the abstention mechanism will always trigger if `anfis_rules` is absent or empty.

4. **Typical call sequence:**
   ```
   GET  /api           → load dataset/model metadata for UI labels
   POST /api/predict   → get prediction + fuzzy rules
   POST /api/rules     → (optional) get full MF parameters for advanced view
   POST /api/copilot   → pass PredictResponse + clinician question
   ```

---

## Project Notes

- **Single worker only** — The server intentionally runs with `--workers 1`. Multiple workers would spawn multiple GPU processes and cause VRAM OOM. The ThreadPoolExecutor with `max_workers=1` in the predict routes enforces serial GPU access.
- **CPU fallback** — If no CUDA device is found, all inference runs on CPU. Performance is lower but functionality is identical.
- **Debug routes** — `app/api/llm_debug.py` and `app/api/rag_debug.py` exist for local development but are **not registered** in `main.py`. Do not add them to production deployments.
- **Offline feature extraction** — The 3D-ResNet10 backbone (MedicalNet) is used only in `predict_scan` for on-the-fly extraction. For large-scale batch processing, run the standalone feature extractor script from the training repository and pass the resulting `.npz` files directly to `/api/predict`.
- **Datasets supported** — OASIS-1 (Alzheimer's), OASIS-2 (longitudinal Alzheimer's), MMIST-ccRCC (kidney cancer), UTSW-Glioma (glioma molecular subtype). Switch datasets by swapping the artifacts as described in [Model Artifacts](#model-artifacts).
