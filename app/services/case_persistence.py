from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Patient, 
    PatientCase, 
    FeatureSet, 
    Scan, 
    Prediction, 
    PredictionScan, 
    VisualEmbedding,
    VisualFinding,
    ANFISRuleSet, 
    FusionResult, 
)

from app.schemas.predict import PredictResponse

@dataclass
class UploadedScanRecord: 
    modality: str
    file_path: str
    file_format: str
    embedding_path: str | None = None

async def persist_prediction_case (
        db: AsyncSession,
        *, 
        patient_first_name: str,
        patient_last_name: str,
        patient_age: int,
        dataset_subject_id: str | None,
        source: str,
        disease_domain: str | None,
        clinical_features: dict[str, Any],
        scans: list[UploadedScanRecord],
        prediction_response: PredictResponse,
) -> dict[str, Any]:
    """
    Persist one completed prediction run.

    Saves:
    - Patient
    - PatientCase
    - Scan
    - FeatureSet
    - Prediction
    - PredictionScan
    - VisualEmbedding
    - FusionResult
    - ANFISRuleSet
    """

    patient = Patient(
        first_name=patient_first_name,
        last_name=patient_last_name,
        age=patient_age,
        dataset_subject_id=dataset_subject_id
    )
    db.add(patient)
    await db.flush()  # To get patient_id for foreign key relationships

    case = PatientCase(
        patient_id=patient.patient_id,
        source=source,
        disease_domain=disease_domain
    )
    db.add(case)
    await db.flush()  # To get case_id for foreign key relationships

    feature_set = FeatureSet(
        case_id=case.case_id,
        features=clinical_features
    )
    db.add(feature_set)
    await db.flush()  # To get feature_set_id for foreign key relationships

    scan_rows: list[Scan] = []

    for scan_input in scans: 
        scan = Scan(
            case_id=case.case_id,
            modality=scan_input.modality,
            file_path=scan_input.file_path,
            file_format=scan_input.file_format
        )
        db.add(scan)
        await db.flush()  # To get scan_id for foreign key relationships

        scan_rows.append(scan)

        if scan_input.embedding_path is not None:
            embedding = VisualEmbedding(
                scan_id=scan.scan_id,
                embedding_path=scan_input.embedding_path,
                embedding_dim=512,
                encoder="MedicalNet ResNet-10",
            )
            db.add(embedding)

    prediction = Prediction(
        case_id=case.case_id,
        feature_set_id=feature_set.feature_set_id,
        predicted_label=str(prediction_response.prediction),
        probability=float(prediction_response.probability),
        threshold=float(prediction_response.threshold),
    )
    db.add(prediction)
    await db.flush()  # To get prediction_id for foreign key relationships

    for scan in scan_rows:
        prediction_scan = PredictionScan(
            prediction_id=prediction.prediction_id,
            scan_id=scan.scan_id,
            modality=scan.modality,
        )
        db.add(prediction_scan)

    fusion_result = FusionResult(
        prediction_id=prediction.prediction_id,
        w_clinical=prediction_response.fusion_weights.w_clinical,
        w_visual=prediction_response.fusion_weights.w_visual,
    )
    db.add(fusion_result)

    for rank, rule in enumerate(prediction_response.anfis_rules, start=1):
        db.add(
            ANFISRuleSet(
                prediction_id=prediction.prediction_id,
                conditions=rule.conditions,
                strength=float(rule.strength),
                rank=rank,
            )
        )
    await db.commit()

    return {
        "patient_id": patient.patient_id,
        "case_id": case.case_id,
        "feature_set_id": feature_set.feature_set_id,
        "prediction_id": prediction.prediction_id,
        "scans": [
            {
                "scan_id": scan.scan_id,
                "modality": scan.modality,
                "file_path": scan.file_path,
                "embedding_path": next(
                    (
                        scan_input.embedding_path
                        for scan_input in scans
                        if scan_input.modality == scan.modality
                    ), None
                ),
            }
            for scan in scan_rows
        ]
    }

def get_file_format(file_path: str) -> str:
    """
    Returns a simple file format string, e.g. "dcm", "nii", "jpg", based on the file extension of the provided file path.
    """
    path = Path(file_path)
    suffixes = path.suffixes  # This will give a list of suffixes, e.g. ['.nii.gz'] or ['.dcm']

    if len(suffixes) >= 2 and suffixes[-2:] == ['.nii', '.gz']:
        return '.nii.gz'
    
    return path.suffix.lower()

async def persist_visual_findings(
        db: AsyncSession,
        *, 
        scan_id: str, 
        prediction_id: str,
        findings: list,
) -> list[VisualFinding]: 
    rows = []

    for finding in findings: 
        row = VisualFinding(
            scan_id=scan_id,
            prediction_id=prediction_id,
            description=finding.description,
            source_model=finding.source_model,
            confidence=finding.confidence
        )
        db.add(row)
        rows.append(row)

    await db.flush()
    await db.commit()

    return rows