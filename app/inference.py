# app/inference.py
import base64, io, numpy as np, torch
from app.schemas import PredictRequest, PredictResponse, FuzzyRule, FusionWeights

def _decode_npz(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded .npz and return the first array."""
    raw = base64.b64decode(b64_str)
    buf = io.BytesIO(raw)
    npz = np.load(buf)
    return npz[npz.files[0]].astype(np.float32)

def _build_fuzzy_rule_strings(
    firing: torch.Tensor,        # (n_rules,)
    feature_names: list[str],
    n_fuzzy_sets: int,
    top_k: int = 3,
) -> list[dict]:
    import itertools
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
    feature_dict : dict[str, float],
    feature_names: list[str],
    train_medians: dict[str, float],
    scaler,
) -> np.ndarray:
    """
    Replicates training preprocessing:
      1. Fill any missing feature with the train-split median (not 0.0).
      2. Apply the fitted StandardScaler.
    """
    raw = np.array(
        [feature_dict.get(f, train_medians.get(f, 0.0)) for f in feature_names],
        dtype=np.float32,
    )
    return scaler.transform(raw.reshape(1, -1)).astype(np.float32)   # (1, n_feat)

def run_single(req: PredictRequest, artifacts: dict) -> PredictResponse:
    model     = artifacts["model"]
    cfg       = artifacts["cfg"]
    scaler    = artifacts["scaler"]
    features  = artifacts["features"]
    threshold = artifacts["threshold"]
    device    = artifacts["device"]

    # ── 1. Clinical features → scaled tensor ─────────────────────────────────
    raw = np.array([req.clinical.features.get(f, 0.0) for f in features],
                   dtype=np.float32)
    scaled = scaler.transform(raw.reshape(1, -1))#.astype(np.float32)   # (1, n_feat)
    scaled = _impute_and_scale(
        req.clinical.features,
        features,
        artifacts["train_medians"],
        scaler,
    )
    clin_t = torch.tensor(scaled, dtype=torch.float32).to(torch.device)

    # ── 2. Imaging modalities → tensors + masks ───────────────────────────────
    batch = {"clinical": clin_t}
    mod_status = {}
    for mcfg in cfg.modalities:
        name = mcfg.name
        if name in req.modalities and req.modalities[name]:
            feat = _decode_npz(req.modalities[name])
            batch[f"mod_{name}"]  = torch.tensor(feat.reshape(1, -1)).to(device)
            batch[f"mask_{name}"] = torch.ones(1).to(device)
            mod_status[name] = "present"
        else:
            batch[f"mod_{name}"]  = torch.zeros(1, mcfg.input_dim).to(device)
            batch[f"mask_{name}"] = torch.zeros(1).to(device)
            mod_status[name] = "missing"

    # ── 3. Forward pass ───────────────────────────────────────────────────────
    with torch.no_grad():
        logit, gate_w, anfis_out = model(batch)
        prob = torch.sigmoid(logit).item()

    gate = gate_w[0].cpu().numpy()           # [w_clinical, w_visual]
    firing = anfis_out["firing"][0]           # (n_rules,)

    rules = _build_fuzzy_rule_strings(
        firing, features, cfg.n_fuzzy_sets, top_k=3
    )

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