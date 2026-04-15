import json, re
from app.services.chroma_store import ChromaStore
from app.services.llm_client import LLMClient


def _word_overlap(text: str, terms: list[str]) -> bool:
    # Word-boundary aware — avoids 'age' falsely matching 'stage'
    words = set(re.findall(r'[a-z_][a-z0-9_]*', text.lower()))
    return any(t.lower() in words for t in terms)


class CopilotService:
    def __init__(self):
        self.store = ChromaStore()
        self.llm = LLMClient()

    def _should_abstain(self, explanation_payload: dict, retrieved: list[dict]) -> tuple[bool, str]:
        if not retrieved:
            return True, "No documents were retrieved from the knowledge base."

        # FIX: use anfis_rules (actual PredictResponse field), not fired_rules/top_features
        anfis_rules = explanation_payload.get("anfis_rules", [])

        # Extract condition keywords from ANFIS rules e.g. "age=HIGH & tumor_size=MED"
        rule_terms = []
        for rule in anfis_rules:
            conditions = rule.get("conditions", "") if isinstance(rule, dict) else getattr(rule, "conditions", "")
            rule_terms += re.findall(r'[a-z_][a-z0-9_]*', conditions.lower())

        retrieved_text = " ".join(item["text"].lower() for item in retrieved)
        if not _word_overlap(retrieved_text, rule_terms):
            return True, "Retrieved docs don't align with model ANFIS rules."
        return False, ""

    def _build_retrieval_query(self, question: str, explanation_payload: dict) -> str:
        # FIX: build query from anfis_rules conditions
        anfis_rules = explanation_payload.get("anfis_rules", [])
        rule_terms = []
        for rule in anfis_rules:
            conditions = rule.get("conditions", "") if isinstance(rule, dict) else getattr(rule, "conditions", "")
            rule_terms += re.findall(r'[a-z_][a-z0-9_]*', conditions.lower())
        parts = [question] + list(dict.fromkeys(rule_terms))  # deduplicated, order-preserving
        return " ".join(parts)

    async def generate_answer(self, question: str, explanation_payload: dict):
        self.store.seed_if_empty()
        retrieved = self.store.query(
            question=self._build_retrieval_query(question, explanation_payload), k=3
        )
        should_abstain, reason = self._should_abstain(explanation_payload, retrieved)

        if should_abstain:
            return json.dumps({
                "summary": "Insufficient grounded evidence to answer confidently.",
                "model_rationale": "Retrieved documents don't align with model explanation.",
                "evidence": [], "citations": [],
                "limitations": reason,
                "uncertainty": "The corpus may be incomplete.",
            }), retrieved

        context_text = "\n\n".join(
            [f"[{item['id']}] {item['text']}" for item in retrieved]
        )

        system_prompt = (
            "You are a medical AI assistant. "
            "You must answer ONLY using the provided context and model explanation. "
            "If the context is insufficient, say you are uncertain."
        )

        user_prompt = f"""
Question:
{question}

Model Output:
{json.dumps(explanation_payload, indent=2)}

Retrieved Context:
{context_text}

Instructions:
- Answer ONLY using the model output and retrieved context.
- If the context is insufficient, say so clearly.
- Return valid JSON only.
- Do not include markdown fences.
- "summary" must be a short string.
- "model_rationale" must be a short string.
- "evidence" must be a list of short bullet strings.
- "citations" must be a list of source IDs like ["rule_r1", "feature_tumor_size"].
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
        # FIX: await the async chat call (was missing await, returning coroutine object)
        raw_response = await self.llm.chat(system_prompt, user_prompt)
        return raw_response, retrieved
