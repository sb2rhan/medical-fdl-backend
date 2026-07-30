from dataclasses import dataclass

@dataclass
class VisualFindingOutput:
    description: str
    source_model: str
    confidence: str

async def generate_visual_findings(
    *,
    scan_path: str, 
    modality: str,
) -> list[VisualFindingOutput]: 
    """
    Temporary placeholder.
    Later this will call a medical VLM.
    """
    return [
        VisualFindingOutput(
            description=f"{modality} scan uploaded and available for clinician review. No VLM findings gnerated yet.",
            source_model="placeholder",
            confidence="N/A"
        )
    ]