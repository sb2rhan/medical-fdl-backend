import base64, io, itertools
import numpy as np
import torch
from app.schemas.predict import PredictRequest, PredictResponse, FuzzyRule, FusionWeights


def _decode_npz(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    buf = io.BytesIO(raw)
    npz = np.load(buf)
    return npz[npz.files[0]].astype(np.float32)


def _build_fuzzy_rule_strings(
    firing: torch.Tensor,
    feature_names: list[str],
    n_fuzzy_sets: int,
    top_k: int = 3,
) -> list[dict]:
    set_labels = ["LOW", "MED", "HIGH"][:n_fuzzy_sets]
    all_rules = list(itertools.product(range(n_fuzzy_sets), repeat=len(feature_names)))
    strengths = firing.detach().cpu().numpy()
    top_idx = strengths.argsort()[::-1][:top_k]
    rules = []
    for idx in top_idx:
        combo = all_rules[idx]
        cond = " & ".join(
            f"{feature_names[i]}={set_labels[combo[i]]}"
            for i in range(len(feature_names))
        )
        rules.append({"conditions": cond, "strength": float(strengths[idx])})
    return rules


def _impute_and_scale(
    feature_dict: dict[str, float],
    feature_names: list[str],
    train_medians: dict[str, float],
    scaler,
) -> np.ndarray:
    """Replicates training preprocessing: median imputation + StandardScaler."""
    raw = np.array(
        [feature_dict.get(f, train_medians.get(f, 0.0)) for f in feature_names],
        dtype=np.float32,
    )
    return scaler.transform(raw.reshape(1, -1)).astype(np.float32)


def run_single(req: PredictRequest, artifacts: dict) -> PredictResponse:
    model             = artifacts["model"]
    modality_configs  = artifacts["modality_configs"]   # list[ModalityConfig]
    n_fuzzy_sets      = artifacts["n_fuzzy_sets"]       # int
    scaler            = artifacts["scaler"]
    features          = artifacts["features"]
    threshold         = artifacts["threshold"]
    device            = artifacts["device"]             # torch.device instance
    calibrator        = artifacts.get("calibrator")    # LogisticRegression | None

    # ── 1. Clinical features → scaled tensor ─────────────────────────────────
    scaled = _impute_and_scale(
        req.clinical.features,
        features,
        artifacts["train_medians"],
        scaler,
    )
    clin_t = torch.tensor(scaled, dtype=torch.float32).to(device, non_blocking=True)

    # ── 2. Imaging modalities → tensors + masks ───────────────────────────────
    batch = {"clinical": clin_t}
    mod_status = {}
    for mcfg in modality_configs:
        name = mcfg.name
        if name in req.modalities and req.modalities[name]:
            feat = _decode_npz(req.modalities[name])
            batch[f"mod_{name}"]  = torch.tensor(feat.reshape(1, -1)).to(device, non_blocking=True)
            batch[f"mask_{name}"] = torch.ones(1).to(device, non_blocking=True)
            mod_status[name] = "present"
        else:
            batch[f"mod_{name}"]  = torch.zeros(1, mcfg.input_dim).to(device, non_blocking=True)
            batch[f"mask_{name}"] = torch.zeros(1).to(device, non_blocking=True)
            mod_status[name] = "missing"

    # ── 3. Forward pass ───────────────────────────────────────────────────────
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logit, gate_w, anfis_out = model(batch)
        logit = logit.float()

    # ── 4. Calibrated probability ─────────────────────────────────────────────
    if calibrator is not None:
        logit_np = logit.detach().cpu().numpy().reshape(-1, 1)
        prob = float(calibrator.predict_proba(logit_np)[0, 1])
    else:
        prob = float(torch.sigmoid(logit).item())

    # ── 5. Assemble response ──────────────────────────────────────────────────
    gate   = gate_w[0].cpu().numpy()
    firing = anfis_out["firing"][0]
    rules  = _build_fuzzy_rule_strings(firing, features, n_fuzzy_sets, top_k=3)

    return PredictResponse(
        subject_id=req.subject_id,
        prediction=int(prob >= threshold),
        probability=round(prob, 4),
        threshold=round(threshold, 4),
        fusion_weights=FusionWeights(
            w_clinical=round(float(gate[0]), 4),
            w_visual=round(float(gate[1]), 4),
        ),
        anfis_rules=[FuzzyRule(**r) for r in rules],
        modality_status=mod_status,
    )
