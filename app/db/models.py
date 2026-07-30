from datetime import datetime, UTC, timezone
from uuid import uuid4

from sqlalchemy import DateTime, String, Float, ForeignKey, Integer, JSON, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

# Utility function to generate unique IDs with a prefix
def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"

class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
        default=lambda: generate_id("patient"),
    )

    # Keeping a dataset subject id for tracking dataset provenance, but it is not required for new external patients 
    dataset_subject_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))

    age: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(UTC), onupdate=datetime.now(UTC))

    cases: Mapped[list["PatientCase"]] = relationship(back_populates="patient")

class PatientCase(Base): 
    __tablename__ = "patient_cases"

    case_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("case"))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"))
    source: Mapped[str] = mapped_column(String(100))
    disease_domain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(UTC), onupdate=datetime.now(UTC))

    patient: Mapped["Patient"] = relationship(back_populates="cases")
    scans: Mapped[list["Scan"]] = relationship(back_populates="case")
    feature_sets: Mapped[list["FeatureSet"]] = relationship(back_populates="case")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="case")

class Scan(Base):
    __tablename__ = "scans"

    scan_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("scan"))
    case_id: Mapped[str] = mapped_column(ForeignKey("patient_cases.case_id"))
    modality: Mapped[str] = mapped_column(String(50))
    file_path: Mapped[str] = mapped_column(String(500)) # TODO: We may want to use Text type for file paths if we expect them to be very long
    file_format: Mapped[str] = mapped_column(String(20))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    case: Mapped["PatientCase"] = relationship(back_populates="scans")
    embeddings: Mapped[list["VisualEmbedding"]] = relationship(back_populates="scan")
    visual_findings: Mapped[list["VisualFinding"]] = relationship(back_populates="scan")
    prediction_scans: Mapped[list["PredictionScan"]] = relationship(back_populates="scan")

class FeatureSet(Base):
    __tablename__ = "feature_sets"

    feature_set_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("feat_set"))
    case_id: Mapped[str] = mapped_column(ForeignKey("patient_cases.case_id"))
    features: Mapped[dict] = mapped_column(JSON)  # Storing clinical features as a JSON object
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    case: Mapped["PatientCase"] = relationship(back_populates="feature_sets")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="feature_set")

class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("pred"))
    case_id: Mapped[str] = mapped_column(ForeignKey("patient_cases.case_id"))
    feature_set_id: Mapped[str] = mapped_column(String(100), ForeignKey("feature_sets.feature_set_id"))
    predicted_label: Mapped[str] = mapped_column(String(100)) # This could be a diagnosis, risk category, or any other type of prediction depending on the use case
    probability: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float) 
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    case: Mapped["PatientCase"] = relationship(back_populates="predictions")
    feature_set: Mapped["FeatureSet"] = relationship(back_populates="predictions")
    fusion_result: Mapped["FusionResult"] = relationship(back_populates="prediction", uselist=False)
    anfis_rules: Mapped[list["ANFISRuleSet"]] = relationship(back_populates="prediction")
    visual_findings: Mapped[list["VisualFinding"]] = relationship(back_populates="prediction") # Not all predictions will have visual findings, so this can be nullable
    evidence_chunks: Mapped[list["EvidenceChunk"]] = relationship(back_populates="prediction")
    prediction_scans: Mapped[list["PredictionScan"]] = relationship(back_populates="prediction")

class PredictionScan(Base):
    __tablename__ = "prediction_scans"

    __table_args__ = (
        UniqueConstraint('prediction_id', 'scan_id', name='uq_prediction_scan'),
    )

    prediction_scan_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("pred_scan"))

    prediction_id: Mapped[str] = mapped_column(String(100), ForeignKey("predictions.prediction_id"))
    scan_id: Mapped[str] = mapped_column(String(100), ForeignKey("scans.scan_id"))

    modality: Mapped[str] = mapped_column(String(50)) # Storing modality here for easier querying, even though it is technically redundant since we can get it through the Scan relationship
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    prediction: Mapped["Prediction"] = relationship(back_populates="prediction_scans")
    scan: Mapped["Scan"] = relationship(back_populates="prediction_scans")

class FusionResult(Base):
    __tablename__ = "fusion_results"

    fusion_result_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("fusion"))
    prediction_id: Mapped[str] = mapped_column(String(100), ForeignKey("predictions.prediction_id"))
    w_clinical: Mapped[float] = mapped_column(Float)
    w_visual: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    prediction: Mapped["Prediction"] = relationship(back_populates="fusion_result")

class ANFISRuleSet(Base):
    __tablename__ = "anfis_rule_sets"

    anfis_rule_set_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("anfis"))
    prediction_id: Mapped[str] = mapped_column(String(100), ForeignKey("predictions.prediction_id"))
    
    conditions: Mapped[str] = mapped_column(Text) 
    strength: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)

    prediction: Mapped["Prediction"] = relationship(back_populates="anfis_rules")
    evidence_chunks: Mapped[list["EvidenceChunk"]] = relationship(back_populates="anfis_rule_set")

class VisualEmbedding(Base):
    __tablename__ = "visual_embeddings"

    embedding_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("embed"))
    scan_id: Mapped[str] = mapped_column(String(100), ForeignKey("scans.scan_id"))
    embedding_path: Mapped[str] = mapped_column(String(500)) 
    embedding_dim: Mapped[int] = mapped_column(Integer, default=512) 
    encoder: Mapped[str] = mapped_column(String(100), default="MedicalNet ResNet-10")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    scan: Mapped["Scan"] = relationship(back_populates="embeddings")

class VisualFinding(Base):
    __tablename__ = "visual_findings"

    finding_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("vf"))
    scan_id: Mapped[str] = mapped_column(String(100), ForeignKey("scans.scan_id"))
    prediction_id: Mapped[str] = mapped_column(String(100), ForeignKey("predictions.prediction_id")) 
    description: Mapped[str] = mapped_column(Text) 
    source_model: Mapped[str | None] = mapped_column(String(100), nullable=True) # VLM model that generated this finding, if applicable
    confidence: Mapped[str | None] = mapped_column(String(50), nullable=True) # Confidence level of the finding, if applicable
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    scan: Mapped["Scan"] = relationship(back_populates="visual_findings")
    prediction: Mapped["Prediction"] = relationship(back_populates="visual_findings")
    evidence_chunks: Mapped[list["EvidenceChunk"]] = relationship(back_populates="visual_finding")

class EvidenceChunk(Base):
    __tablename__ = "evidence_chunks"

    chunk_id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: generate_id("chunk"))
    chroma_doc_id: Mapped[str] = mapped_column(String(200), unique=True)
    prediction_id: Mapped[str] = mapped_column(String(100), ForeignKey("predictions.prediction_id"))
    anfis_rule_set_id: Mapped[str | None] = mapped_column(String(100), ForeignKey("anfis_rule_sets.anfis_rule_set_id"), nullable=True)
    visual_finding_id: Mapped[str | None] = mapped_column(String(100), ForeignKey("visual_findings.finding_id"), nullable=True)

    evidence_type: Mapped[str] = mapped_column(String(50))
    source_table: Mapped[str] = mapped_column(String(100))
    source_id: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    prediction: Mapped["Prediction"] = relationship(back_populates="evidence_chunks")
    anfis_rule_set: Mapped["ANFISRuleSet"] = relationship(back_populates="evidence_chunks")
    visual_finding: Mapped["VisualFinding"] = relationship(back_populates="evidence_chunks")








