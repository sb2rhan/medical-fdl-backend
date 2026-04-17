def get_seed_documents():
    return [
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
            "id": "rule_mmse_low_nwbv_low",
            "text": (
                "Model interpretation note: a rule combining low MMSE and low nWBV may push the model "
                "toward a higher-risk prediction, especially when supported by other clinical factors."
            ),
            "metadata": {
                "source": "rulebook",
                "title": "Low MMSE and Low nWBV Rule",
                "chunk_id": "rule_mmse_low_nwbv_low_chunk_1"
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
    ]