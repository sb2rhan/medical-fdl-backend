from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession 

from app.db.models import EvidenceChunk, VisualFinding
from app.schemas.predict import PredictResponse
from app.services.chroma_store import ChromaStore

def _new_chroma_id(prefix: str) -> str:
    """Generate a new Chroma ID."""
    return f"{prefix}_{uuid4().hex}"

async def index_prediction_evidence(
        db: AsyncSession, 
        *, 
        subject_id: str, 
        response: PredictResponse
) -> list[str]: 
    """
    Create patient-specific evidence chunks in ChromaDB and link them in PostgreSQL.
    Requires response.persistence to be present.
    """

    if response.persistence is None: 
        return []
    
    store = ChromaStore()
    created_doc_ids: list[str] = []

    prediction_id = response.persistence.prediction_id
    case_id = response.persistence.case_id

    # 1. Prediction summary chunk
    prediction_doc_id = _new_chroma_id("ev_pred")
    prediction_text = (
        f"Patient case {case_id} has prediction {prediction_id}"
        f"The model predicted label {response.prediction} with probability"
        f"{response.probability} using threshold {response.threshold}"
    )

    store.add_document(
        doc_id=prediction_doc_id, 
        text=prediction_text,
        metadata={
            "subject_id": subject_id,
            "case_id": case_id,
            "prediction_id": prediction_id,
            "evidence_type": "prediction_summary",
            "source_table": "predictions",
            "source_id": prediction_id
        }
    )

    db.add(
        EvidenceChunk(
            chroma_doc_id=prediction_doc_id,
            prediction_id=prediction_id,
            evidence_type="prediction_summary",
            source_table="predictions",
            source_id=prediction_id
        )
    )

    created_doc_ids.append(prediction_doc_id)

    # 2. Fusion weights chunk
    fusion_doc_id = _new_chroma_id("ev_fusion")
    fusion_text = (
        f"For prediction {prediction_id}, the clinical branch weight was "
        f"{response.fusion_weights.w_clinical}, and the visual branch weight was "
        f"{response.fusion_weights.w_visual}."
    )

    store.add_document(
        doc_id=fusion_doc_id,
        text=fusion_text,
        metadata={
            "subject_id": subject_id,
            "case_id": case_id,
            "prediction_id": prediction_id,
            "evidence_type": "fusion_summary",
            "source_table": "fusion_results",
            "source_id": prediction_id
        }
    )

    db.add(
        EvidenceChunk(
            chroma_doc_id=fusion_doc_id,
            prediction_id=prediction_id,
            evidence_type="fusion_summary",
            source_table="fusion_results",
            source_id=prediction_id
        )
    )

    created_doc_ids.append(fusion_doc_id)

     # 3. Scan reference chunks
    for scan in response.persistence.scans:
        scan_doc_id = _new_chroma_id("ev_scan")
        scan_text = (
            f"Prediction {prediction_id} used {scan.modality} scan {scan.scan_id}. "
            f"The scan file is stored at {scan.file_path}. "
            f"The extracted MedicalNet embedding is stored at {scan.embedding_path}."
        )

        store.add_document(
            doc_id=scan_doc_id,
            text=scan_text,
            metadata={
                "subject_id": subject_id,
                "case_id": case_id,
                "prediction_id": prediction_id,
                "scan_id": scan.scan_id,
                "modality": scan.modality,
                "evidence_type": "scan_reference",
                "source_table": "scans",
                "source_id": scan.scan_id,
            },
        )

        db.add(
            EvidenceChunk(
                chroma_doc_id=scan_doc_id,
                prediction_id=prediction_id,
                evidence_type="scan_reference",
                source_table="scans",
                source_id=scan.scan_id,
            )
        )
        created_doc_ids.append(scan_doc_id)

    # 4. ANFIS rule chunks
    for rank, rule in enumerate(response.anfis_rules, start=1):
        rule_doc_id = _new_chroma_id("ev_rule")
        rule_text = (
            f"For prediction {prediction_id}, ANFIS rule rank {rank} was: "
            f"{rule.conditions}. "
            f"The rule firing strength was {rule.strength}."
        )

        store.add_document(
            doc_id=rule_doc_id,
            text=rule_text,
            metadata={
                "subject_id": subject_id,
                "case_id": case_id,
                "prediction_id": prediction_id,
                "rank": rank,
                "evidence_type": "anfis_rule_result",
                "source_table": "anfis_rule_sets",
                "source_id": rule_doc_id,
            },
        )

        db.add(
            EvidenceChunk(
                chroma_doc_id=rule_doc_id,
                prediction_id=prediction_id,
                evidence_type="anfis_rule_result",
                source_table="anfis_rule_sets",
                source_id=rule_doc_id,
            )
        )
        created_doc_ids.append(rule_doc_id)

    await db.commit()
    return created_doc_ids

async def index_visual_findings (
        db: AsyncSession,
        *, 
        subject_id: str,
        case_id: str,
        prediction_id: str,
        visual_findings: list[VisualFinding]
) -> list[str]:
    """
    Index saved visual findings into ChromaDB and link them in PostgreSQL.
    """

    store = ChromaStore()
    created_doc_ids: list[str] = []

    for finding in visual_findings:
        doc_id = _new_chroma_id("ev_visual")

        text = (
            f"Visual finding for prediction {prediction_id}, scan {finding.scan_id}: "
            f"{finding.description} "
            f"Source model: {finding.source_model or 'Unknown'}."
            f"Confidence: {finding.confidence or 'Unknown'}."
        )

        store.add_document(
            doc_id=doc_id,
            text=text,
            metadata={
                "subject_id": subject_id,
                "case_id": case_id,
                "prediction_id": prediction_id,
                "scan_id": finding.scan_id,
                "evidence_type": "visual_finding",
                "source_table": "visual_findings",
                "source_id": finding.finding_id,
            },
        )

        db.add(
            EvidenceChunk(
                chroma_doc_id=doc_id,
                prediction_id=prediction_id,
                evidence_type="visual_finding",
                source_table="visual_findings",
                source_id=finding.finding_id,
            )
        )
        created_doc_ids.append(doc_id)

        await db.commit()
        return created_doc_ids
