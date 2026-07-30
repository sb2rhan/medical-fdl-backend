from itertools import product


FEATURES = ["MMSE", "nWBV", "Age", "Educ"]
FUZZY_LABELS = ["LOW", "MED"]


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def _generate_rule_documents():
    docs = []

    for labels in product(FUZZY_LABELS, repeat=len(FEATURES)):
        rule_parts = [f"{feature}={label}" for feature, label in zip(FEATURES, labels)]
        rule_text = " & ".join(rule_parts)

        rule_id = "rule_" + "_".join(
            f"{_slug(feature)}_{_slug(label)}"
            for feature, label in zip(FEATURES, labels)
        )

        readable_conditions = ", ".join(
            f"{feature} is {label}" for feature, label in zip(FEATURES, labels)
        )

        docs.append({
            "id": rule_id,
            "text": (
                f"ANFIS fuzzy rule: {rule_text}. "
                f"This rule activates when {readable_conditions}. "
                "The rule combines cognitive score, normalized brain volume, age, and education level. "
                "Its actual contribution depends on the patient-specific firing strength returned by the model. "
                "Rules with stronger firing strength have greater influence on the clinical explanation."
            ),
            "metadata": {
                "source": "anfis_rulebook",
                "title": f"ANFIS Rule: {rule_text}",
                "chunk_id": f"{rule_id}_chunk_1",
                "rule_conditions": rule_text,
            },
        })

    return docs


def get_seed_documents():
    base_docs = [
        {
            "id": "feature_mmse",
            "text": (
                "Feature dictionary: MMSE refers to the Mini-Mental State Examination score. "
                "Lower MMSE values may reflect greater cognitive impairment and can contribute "
                "to a higher-risk model prediction."
            ),
            "metadata": {
                "source": "feature_dictionary",
                "title": "MMSE Feature",
                "chunk_id": "feature_mmse_chunk_1"
            },
        },
        {
            "id": "feature_nwbv",
            "text": (
                "Feature dictionary: nWBV refers to normalized whole brain volume. "
                "Lower nWBV values may indicate greater brain atrophy and can contribute "
                "to a higher-risk model prediction."
            ),
            "metadata": {
                "source": "feature_dictionary",
                "title": "nWBV Feature",
                "chunk_id": "feature_nwbv_chunk_1"
            },
        },
        {
            "id": "feature_age",
            "text": (
                "Feature dictionary: Age is the patient's age in years. "
                "Age may contribute to model risk estimation but should not be interpreted alone."
            ),
            "metadata": {
                "source": "feature_dictionary",
                "title": "Age Feature",
                "chunk_id": "feature_age_chunk_1"
            },
        },
        {
            "id": "feature_educ",
            "text": (
                "Feature dictionary: Educ refers to years or level of education used in the model input. "
                "It may influence risk estimation together with the other clinical features."
            ),
            "metadata": {
                "source": "feature_dictionary",
                "title": "Education Feature",
                "chunk_id": "feature_educ_chunk_1"
            },
        },
        {
            "id": "rule_clinical_only",
            "text": (
                "Model interpretation note: if MRI is missing, the model can fall back to clinical-only "
                "reasoning. In that case, the explanation should be interpreted as based on clinical input "
                "without imaging support."
            ),
            "metadata": {
                "source": "rulebook",
                "title": "Clinical-only Fallback",
                "chunk_id": "rule_clinical_only_chunk_1"
            },
        },
        {
            "id": "guideline_missing_modality",
            "text": (
                "Guideline excerpt: missing imaging data can reduce the completeness of a multimodal model "
                "explanation. Predictions based only on clinical features may still be useful, but they "
                "should be interpreted with added caution."
            ),
            "metadata": {
                "source": "guideline_excerpt",
                "title": "Missing Modality Guidance",
                "chunk_id": "guideline_missing_modality_chunk_1"
            },
        },
        {
            "id": "guideline_threshold",
            "text": (
                "Guideline excerpt: when a model probability is close to the decision threshold, the result "
                "should be treated as borderline rather than definitive."
            ),
            "metadata": {
                "source": "guideline_excerpt",
                "title": "Threshold Interpretation Note",
                "chunk_id": "guideline_threshold_chunk_1"
            },
        },
        {
            "id": "guideline_firing_strength",
            "text": (
                "ANFIS interpretation note: firing strength measures how strongly a fuzzy rule is activated "
                "for a specific patient. A higher firing strength means the rule is more relevant to the "
                "current prediction explanation."
            ),
            "metadata": {
                "source": "guideline_excerpt",
                "title": "Firing Strength Interpretation",
                "chunk_id": "guideline_firing_strength_chunk_1"
            },
        },
        {
            "id": "guideline_fusion_weights",
            "text": (
                "Multimodal fusion note: fusion weights describe how much the final prediction relied on "
                "clinical information versus visual imaging information. If MRI is missing, the visual weight "
                "is zero and the clinical weight becomes one."
            ),
            "metadata": {
                "source": "guideline_excerpt",
                "title": "Fusion Weight Interpretation",
                "chunk_id": "guideline_fusion_weights_chunk_1"
            },
        },
    ]

    return base_docs + _generate_rule_documents()