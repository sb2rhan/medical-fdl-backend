from pydantic import BaseModel, Field
from typing import Optional


class TopFeature(BaseModel):
    name: str
    weight: float


# ── Single-sample input ───────────────────────────────────────────────────────────
class ClinicalFeatures(BaseModel):
    """Key-value map of clinical feature name → float value."""
    features: dict[str, float]


class PredictRequest(BaseModel):
    subject_id: str
    clinical: ClinicalFeatures
    # base64-encoded .npz bytes per modality, e.g. {"MRI": "<b64>", "CT": "<b64>"}
    modalities: dict[str, str] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    samples: list[PredictRequest]


# ── Outputs ───────────────────────────────────────────────────────────────────
class FuzzyRule(BaseModel):
    """Active ANFIS rule schema. conditions e.g. 'MMSE=LOW & nWBV=LOW'."""
    conditions: str
    strength: float


class FusionWeights(BaseModel):
    w_clinical: float
    w_visual: float


class PredictResponse(BaseModel):
    subject_id: str
    prediction: int
    probability: float           # Platt-calibrated when calibrator is loaded
    threshold: float             # Youden's J threshold from checkpoint
    fusion_weights: FusionWeights
    anfis_rules: list[FuzzyRule]
    modality_status: dict[str, str]  # {"MRI": "present", "CT": "missing"}


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class MetadataResponse(BaseModel):
    dataset_name:       str
    modalities:         list[str]
    clinical_features:  list[str]
    n_fuzzy_sets:       int
    threshold:          float
    target_names:       list[str]
    device:             str
    best_epoch:         int
    val_bacc:           float
    val_auc:            float
    calibrator_loaded:  bool     # True when platt_calibrator.pkl was found
