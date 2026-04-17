import json, re
from app.services.chroma_store import ChromaStore
from app.services.llm_client import LLMClient


def _word_overlap(text: str, terms: list[str]) -> bool:
    # Word-boundary aware — avoids 'age' falsely matching 'stage'
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


class CopilotService:
    def __init__(self):
        self.store = ChromaStore()
        self.llm = LLMClient()

    def _payload_is_sufficient(self, question: str, explanation_payload: dict) -> bool:
        """
        Returns True when the question can be answered mainly from the model
        explanation payload, even if retrieval is weak.
        """
        q = question.lower()

        payload_keywords = [
            "why",
            "explain",
            "prediction",
            "predicted",
            "factor",
            "factors",
            "risk",
            "probability",
            "threshold",
            "rule",
            "rules",
            "anfis",
            "mean",
            "means",
            "reliable",
            "reliability",
            "missing",
            "mri",
            "modality",
            "clinical",
        ]

        # minimal payload evidence required
        has_prediction = "prediction" in explanation_payload
        has_probability = "probability" in explanation_payload
        has_rules = bool(explanation_payload.get("anfis_rules"))
        has_modalities = bool(explanation_payload.get("modality_status"))

        asks_payload_question = any(k in q for k in payload_keywords)

        return asks_payload_question and (has_prediction or has_probability or has_rules or has_modalities)

    def _is_domain_mismatch(self, question: str, explanation_payload: dict) -> bool:
        """
        Detect clearly mismatched questions that should still abstain even if
        a payload exists.
        """
        q = question.lower()
        rule_terms = _extract_rule_terms(explanation_payload)

        # current payload domain is cognitive/OASIS-style if these features exist
        cognitive_terms = {"mmse", "nwbv", "educ", "age"}
        has_cognitive_payload = any(term in rule_terms for term in cognitive_terms)

        kidney_ct_terms = {
            "kidney", "renal", "tumor", "staging", "stage", "ct", "lesion", "mass"
        }

        if has_cognitive_payload and any(term in q for term in kidney_ct_terms):
            return True

        return False

    def _should_abstain(self, question: str, explanation_payload: dict, retrieved: list[dict]) -> tuple[bool, str]:
        """
        Abstain only when:
        1) there is a clear domain mismatch, or
        2) retrieval is absent/misaligned AND the payload itself is not sufficient.
        """
        if self._is_domain_mismatch(question, explanation_payload):
            return True, "The question does not align with the provided model explanation and patient context."

        payload_sufficient = self._payload_is_sufficient(question, explanation_payload)

        if not retrieved:
            if payload_sufficient:
                return False, ""
            return True, "No documents were retrieved from the knowledge base."

        rule_terms = _extract_rule_terms(explanation_payload)
        retrieved_text = " ".join(item["text"].lower() for item in retrieved)

        if not _word_overlap(retrieved_text, rule_terms):
            if payload_sufficient:
                return False, ""
            return True, "Retrieved docs don't align with model ANFIS rules."

        return False, ""

    def _build_retrieval_query(self, question: str, explanation_payload: dict) -> str:
        rule_terms = _extract_rule_terms(explanation_payload)
        parts = [question] + rule_terms
        return " ".join(parts)

    async def generate_answer(self, question: str, explanation_payload: dict):
        self.store.seed_if_empty()

        retrieved = self.store.query(
            question=self._build_retrieval_query(question, explanation_payload),
            k=3
        )

        # For debugging retrieval relevance during development:
        print("Retrieved IDs:", [item["id"] for item in retrieved])

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
            [f"[{item['id']}] {item['text']}" for item in retrieved]
        ) if retrieved else "No supporting documents were retrieved."

        system_prompt = (
            "You are a medical AI assistant. "
            "Use the explanation payload as the primary source of truth for this patient-specific answer. "
            "Use retrieved context only to support, clarify, or qualify the explanation. "
            "If the question asks for information outside the payload and context, say you are uncertain. "
            "Do not invent facts."
        )

        user_prompt = f"""
Question:
{question}

Model Output (primary source of truth):
{json.dumps(explanation_payload, indent=2)}

Retrieved Context (secondary support):
{context_text}

Instructions:
- Answer using the model output first, and retrieved context second.
- If retrieved context is weak but the model output is sufficient, still answer cautiously.
- If the question goes beyond both the model output and context, say so clearly.
- Return valid JSON only.
- Do not include markdown fences.
- "summary" must be a short string.
- "model_rationale" must be a short string.
- "evidence" must be a list of short bullet strings.
- "citations" must be a list of source IDs like ["rule_r1", "feature_age"].
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