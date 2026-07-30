from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.predict import PredictResponse
from app.services.visual_explainer import generate_visual_findings
from app.services.case_persistence import persist_visual_findings
from app.services.evidence_indexer import (
    index_prediction_evidence,
    index_visual_findings,
)


async def enrich_prediction_evidence(
    *,
    db: AsyncSession,
    subject_id: str,
    result: PredictResponse,
) -> None:
    """
    Index prediction evidence, generate visual findings, persist them,
    and index visual finding evidence.
    """

    if result.persistence is None:
        raise ValueError("Cannot enrich prediction evidence without persistence metadata.")

    chroma_doc_ids = await index_prediction_evidence(
        db=db,
        subject_id=subject_id,
        response=result,
    )
    print(f"Indexed prediction evidence with IDs: {chroma_doc_ids}")

    for scan_ref in result.persistence.scans:
        findings = await generate_visual_findings(
            scan_path=scan_ref.file_path,
            modality=scan_ref.modality,
        )

        finding_rows = await persist_visual_findings(
            db=db,
            scan_id=scan_ref.scan_id,
            prediction_id=result.persistence.prediction_id,
            findings=findings,
        )

        visual_doc_ids = await index_visual_findings(
            db=db,
            subject_id=subject_id,
            case_id=result.persistence.case_id,
            prediction_id=result.persistence.prediction_id,
            visual_findings=finding_rows,
        )

        print(f"Indexed visual findings with IDs: {visual_doc_ids}")