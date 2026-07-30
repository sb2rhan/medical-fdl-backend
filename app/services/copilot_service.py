import json
import re

from app.services.chroma_store import ChromaStore
from app.services.llm_client import LLMClient


def _word_overlap(text: str, terms: list[str]) -> bool:
    words = set(re.findall(r"[a-z_][a-z0-9_]*", text.lower()))
    return any(t.lower() in words for t in terms)


def _extract_rule_terms(explanation_payload: dict) -> list[str]:
    anfis_rules = explanation_payload.get("anfis_rules", [])
    rule_terms = []
    ignore = {"low", "med", "high", "and"}

    for rule in anfis_rules:
        conditions = rule.get("conditions", "") if isinstance(rule, dict) else getattr(rule, "conditions", "")
        terms = re.findall(r"[a-z_][a-z0-9_]*", conditions.lower())
        rule_terms += [t for t in terms if t not in ignore]

    return list(dict.fromkeys(rule_terms))


def _extract_payload_ids(explanation_payload: dict) -> list[str]:
    persistence = explanation_payload.get("persistence") or {}
    ids = []

    for key in ["patient_id", "case_id", "feature_set_id", "prediction_id"]:
        value = persistence.get(key)
        if value:
            ids.append(str(value))

    for scan in persistence.get("scans", []) or []:
        if scan.get("scan_id"):
            ids.append(str(scan["scan_id"]))
        if scan.get("modality"):
            ids.append(str(scan["modality"]))

    return ids


def _extract_modality_terms(explanation_payload: dict) -> list[str]:
    terms = []

    modality_status = explanation_payload.get("modality_status", {})
    for modality, status in modality_status.items():
        terms.append(str(modality))
        terms.append(str(status))

    persistence = explanation_payload.get("persistence") or {}
    for scan in persistence.get("scans", []) or []:
        terms.append(str(scan.get("modality", "")))
        terms.append(str(scan.get("scan_id", "")))
        terms.append(str(scan.get("file_path", "")))
        terms.append(str(scan.get("embedding_path", "")))

    return [t for t in terms if t]


def _extract_rule_texts(explanation_payload: dict) -> list[str]:
    rules = explanation_payload.get("anfis_rules", [])
    output = []

    for rule in rules:
        conditions = rule.get("conditions", "") if isinstance(rule, dict) else getattr(rule, "conditions", "")
        strength = rule.get("strength", "") if isinstance(rule, dict) else getattr(rule, "strength", "")
        if conditions:
            output.append(f"{conditions} strength {strength}")

    return output


class CopilotService:
    def __init__(self):
        self.store = ChromaStore()
        self.llm = LLMClient()

    def _payload_is_sufficient(self, question: str, explanation_payload: dict) -> bool:
        q = question.lower()

        payload_keywords = [
            "why", "explain", "prediction", "predicted",
            "factor", "factors", "risk", "probability", "threshold",
            "rule", "rules", "anfis", "mean", "means",
            "reliable", "reliability", "missing",
            "mri", "ct", "pet", "scan", "image", "modality",
            "clinical", "visual", "fusion", "weight", "weights",
            "strongest", "used",
        ]

        has_prediction = "prediction" in explanation_payload
        has_probability = "probability" in explanation_payload
        has_rules = bool(explanation_payload.get("anfis_rules"))
        has_modalities = bool(explanation_payload.get("modality_status"))
        has_persistence = bool(explanation_payload.get("persistence"))

        asks_payload_question = any(k in q for k in payload_keywords)

        return asks_payload_question and (
            has_prediction or has_probability or has_rules or has_modalities or has_persistence
        )

    def _is_domain_mismatch(self, question: str, explanation_payload: dict) -> bool:
        q_words = set(re.findall(r"[a-z_][a-z0-9_]*", question.lower()))
        rule_terms = set(_extract_rule_terms(explanation_payload))

        cognitive_terms = {"mmse", "nwbv", "educ", "age"}
        has_cognitive_payload = any(term in rule_terms for term in cognitive_terms)

        kidney_ct_terms = {
            "kidney", "renal", "tumor", "staging", "stage", "lesion", "mass"
        }

        strong_off_domain = {"kidney", "renal", "tumor", "staging", "lesion", "mass"}
        matched_off_domain = q_words.intersection(kidney_ct_terms)

        if has_cognitive_payload:
            if q_words.intersection(strong_off_domain):
                return True
            if len(matched_off_domain) >= 2:
                return True

        return False

    def _should_abstain(
        self,
        question: str,
        explanation_payload: dict,
        retrieved: list[dict],
    ) -> tuple[bool, str]:
        if self._is_domain_mismatch(question, explanation_payload):
            return True, "The question does not align with the provided model explanation and patient context."

        payload_sufficient = self._payload_is_sufficient(question, explanation_payload)

        if not retrieved:
            if payload_sufficient:
                return False, ""
            return True, "No documents were retrieved from the knowledge base."

        rule_terms = _extract_rule_terms(explanation_payload)
        modality_terms = _extract_modality_terms(explanation_payload)
        id_terms = _extract_payload_ids(explanation_payload)

        alignment_terms = rule_terms + modality_terms + id_terms
        retrieved_text = " ".join(item["text"].lower() for item in retrieved)

        if alignment_terms and not _word_overlap(retrieved_text, alignment_terms):
            if payload_sufficient:
                return False, ""
            return True, "Retrieved docs do not align with the model explanation, case IDs, modality, or ANFIS rules."

        return False, ""

    def _build_retrieval_query(self, question: str, explanation_payload: dict) -> str:
        parts = [
            question,
            *(_extract_payload_ids(explanation_payload)),
            *(_extract_rule_terms(explanation_payload)),
            *(_extract_rule_texts(explanation_payload)),
            *(_extract_modality_terms(explanation_payload)),
        ]

        return " ".join(str(p) for p in parts if p)

    async def generate_answer(self, question: str, explanation_payload: dict):
        self.store.seed_if_empty()

        retrieval_query = self._build_retrieval_query(question, explanation_payload)

        retrieved = self.store.query(
            question=retrieval_query,
            k=5,
        )

        should_abstain, reason = self._should_abstain(question, explanation_payload, retrieved)

        if should_abstain:
            return json.dumps({
                "summary": "Insufficient grounded evidence to answer confidently.",
                "model_rationale": "The question could not be safely grounded in the provided explanation and retrieved context.",
                "evidence": [],
                "citations": [],
                "limitations": reason,
                "uncertainty": "The corpus or question may not match the provided case.",
            }), retrieved

        context_text = "\n\n".join(
            [
                f"[{item['id']}]\n"
                f"Metadata: {json.dumps(item.get('metadata', {}))}\n"
                f"Text: {item['text']}"
                for item in retrieved
            ]
        ) if retrieved else "No supporting documents were retrieved."

        system_prompt = (
            "You are a medical AI assistant. "
            "Use the explanation payload as the primary source of truth for patient-specific model outputs. "
            "Use retrieved context to support, cite, and link the answer to stored evidence. "
            "Prefer patient-specific retrieved chunks when they are relevant. "
            "Do not invent facts."
        )

        user_prompt = f"""
Question:
{question}

Model Output / Explanation Payload:
{json.dumps(explanation_payload, indent=2)}

Retrieved Context:
{context_text}

Instructions:
- Answer using the model output first.
- Use retrieved context to support the answer and provide evidence IDs.
- Prefer citing patient-specific retrieved chunks such as ev_pred, ev_scan, ev_fusion, or ev_rule when relevant.
- If answering about scans/images, include scan_id and modality when available.
- If answering about ANFIS rules, identify the strongest rule by highest strength.
- If retrieved context is not directly relevant, answer cautiously. If retrieved patient-specific evidence directly supports the answer, do not describe it as weak.
- If the question goes beyond both model output and retrieved context, say so clearly.
- Do not claim that a visual finding caused the prediction unless that is explicitly supported.
- Return valid JSON only.
- Do not include markdown fences.
- "summary" must be a short string.
- "model_rationale" must be a short string.
- "evidence" must be a list of short strings.
- "citations" must be a list of source IDs from the model payload or retrieved context.
- "limitations" must be a short string.
- "uncertainty" must be a short string.

Return this exact JSON shape:
{{
  "summary": "string",
  "model_rationale": "string",
  "evidence": ["string", "string"],
  "citations": ["string", "string"],
  "limitations": "string",
  "uncertainty": "string"
}}
"""
        raw_response = await self.llm.chat(system_prompt, user_prompt)
        return raw_response, retrieved