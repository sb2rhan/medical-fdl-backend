from typing import Callable, Optional
from dataclasses import dataclass, field

import torch
import numpy as np
import os
import re

def load_npz(filepath: str, feature_dim: int) -> torch.Tensor:
    """Load a pre-extracted .npz feature vector -> FloatTensor (feature_dim,)."""
    data = np.load(filepath)
    vec = data[data.files[0]].flatten()[:feature_dim]
    if len(vec) < feature_dim:
        vec = np.pad(vec, (0, feature_dim - len(vec)))
    return torch.tensor(vec, dtype=torch.float32)


def extract_oasis_id(filepath: str) -> Optional[str]:
    """Extract 'OAS1_XXXX_MR[1|2]' from OASIS feature filenames."""
    match = re.search(r"(OAS1_\d{4}_MR[12])", os.path.basename(filepath))
    return match.group(1) if match else None

@dataclass
class ModalityConfig:
    """
    Describes a single imaging modality.
    All modalities are pre-extracted .npz flat feature vectors.
    Raw image processing (NII, DICOM) is handled by extract_features.py,
    which runs offline once before training.

    Fields
    ------
    name            : unique key used in batch dicts, e.g. "MRI", "CT", "WSI"
    input_dim       : feature vector length (output dim of the extractor)
    latent_dim      : embedding size after FlatEncoder projection
    loader_fn       : optional custom loader override callable(filepath) → Tensor
    lookup_csv      : optional CSV mapping subject_id → npz filename (MMIST-style)
    lookup_id_col   : subject ID column in lookup_csv
    lookup_file_col : filename column in lookup_csv
    """
    name            : str
    input_dim       : int
    latent_dim      : int = 8
    loader_fn       : Optional[Callable] = None
    lookup_csv      : Optional[str] = None
    lookup_id_col   : str = "case_id"
    lookup_file_col : str = "chosen_exam"


@dataclass
class DatasetConfig:
    """
    Top-level config for any multimodal dataset.
    Add a new dataset by defining a new instance — no other code changes needed.
    """
    name              : str
    modalities        : list[ModalityConfig]

    # Clinical
    clinical_csv      : str
    subject_id_col    : str
    label_col         : str
    clinical_features : list[str]
    target_names      : list[str]
    label_transform   : Optional[Callable] = None

    # Modality folders {modality name → folder of .npz files}
    modality_folders  : dict = field(default_factory=dict)

    # per-dataset id extractor (replaces hardcoded OASIS branch)
    id_extractor      : Optional[Callable] = None

    # Training hyperparams
    n_fuzzy_sets      : int   = 2
    n_selected_feats  : int   = 4
    batch_size        : int   = 16
    num_epochs        : int   = 150
    lr                : float = 3e-4
    weight_decay      : float = 0.05
    train_split       : float = 0.7
    val_split         : float = 0.15
    # Loss hyperparameters
    focal_gamma       : float = 2.0
    focal_smoothing   : float = 0.05
    # Files
    model_path        : str   = "best_model.pth"


# ─────────────────────────────────────────────────────────────────────────────
# OASIS Config
# Pre-extracted MRI features: extract_features.py → MRI_features/*.npz (dim=512)
# ─────────────────────────────────────────────────────────────────────────────
OASIS_ROOT = "/content/drive/MyDrive/CSCI408 409: Senior Project/Datasets/oasis"

OASIS_CONFIG = DatasetConfig(
    name              = "OASIS",
    id_extractor      = extract_oasis_id,
    modalities        = [
        ModalityConfig(
            name      = "MRI",
            input_dim = 512,      # must match FeatureExtractor output dim
            latent_dim = 16,       # smaller latent — 416 subjects dataset
        ),
    ],
    modality_folders  = {"MRI": f"{OASIS_ROOT}/features"},
    clinical_csv      = f"{OASIS_ROOT}/oasis_cross-sectional.csv",
    subject_id_col    = "ID",
    label_col         = "CDR",
    clinical_features = ["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF"],
    target_names = ["No Dementia", "Dementia"],
    label_transform   = lambda cdr: 1.0 if float(cdr) > 0 else 0.0,
    n_fuzzy_sets      = 2,
    n_selected_feats  = 5,
    batch_size        = 16,
    num_epochs        = 150,
    lr                = 3e-4,
    weight_decay      = 0.05,
    train_split       = 0.7,
    val_split         = 0.15,
    model_path=f"{OASIS_ROOT}/models/best_oasis_model.pt",
)


# ─────────────────────────────────────────────────────────────────────────────
# MMIST Config  (add back easily with flat encoders + 3 modalities)
# ─────────────────────────────────────────────────────────────────────────────

MMIST_ROOT = "/content/drive/MyDrive/CSCI408 409: Senior Project/Datasets/mmist_data"
MMIST_CONFIG = DatasetConfig(
    name         = "MMIST",
    modalities = [
        ModalityConfig(
            name="CT", input_dim=512,  latent_dim=16,
            lookup_csv      = f"{MMIST_ROOT}/CT_Merged.csv",
            lookup_id_col   = "case_id",
            lookup_file_col = "chosen_exam",
        ),
        ModalityConfig(
            name="MRI", input_dim=512,  latent_dim=16,
            lookup_csv      = f"{MMIST_ROOT}/MRI_Merged.csv",
            lookup_id_col   = "case_id",
            lookup_file_col = "chosen_exam",
        ),
        ModalityConfig(
            name="WSI", input_dim=2048, latent_dim=16,
            lookup_csv      = f"{MMIST_ROOT}/WSI_patientfiles.csv",
            lookup_id_col   = "case_id",
            lookup_file_col = "chosen_exam",
        ),
    ],
    modality_folders  = {
        "CT"  : f"{MMIST_ROOT}/CT_features",
        "MRI" : f"{MMIST_ROOT}/MRI_features",
        "WSI" : f"{MMIST_ROOT}/WSI_features",
    },
    clinical_csv      = f"{MMIST_ROOT}/clinical+genomic_split.csv",
    subject_id_col    = "case_id",
    label_col         = "vital_status_12",
    clinical_features = ['gender','age_diag','grade','ajcc_path_tumor_pt',
                         'ajcc_path_nodes_pn','ajcc_clin_metastasis_cm',
                         'ajcc_path_metastasis_pm','ajcc_path_tumor_stage',
                         'race_Asian','race_Black or African American',
                         'race_Hispanic or Latino','race_White','race_other',
                         'VHL_mutation','PBMR1_mutation','TTN_mutation'],
    target_names = ["Deceased", "Alive"],
    label_transform   = None,
    n_fuzzy_sets      = 2,
    n_selected_feats  = 5,
    batch_size        = 16,
    num_epochs        = 150,
    lr                = 3e-4,
    weight_decay      = 0.05,
    train_split       = 0.6,
    val_split         = 0.2,
    model_path        = f"{MMIST_ROOT}/models/best_mmist_model.pt",
)


