import json
import math
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INTENTS = {"acesso", "rede", "hardware", "impressora"}
DEFAULT_THRESHOLD = 0.35
DEFAULT_AMBIGUITY_MARGIN = 0.05
FALLBACK_ANSWER = (
    "Não reconheci uma orientação confiável para essa mensagem. "
    "Reformule descrevendo um problema de TI ou procure o suporte humano "
    "simulado. Nenhum chamado foi aberto."
)
AMBIGUOUS_ANSWER = (
    "Encontrei possibilidades em categorias diferentes. "
    "Descreva um problema por vez e indique o equipamento ou acesso afetado."
)


class ChatbotEngine:

    def __init__(self, knowledge_path, threshold=DEFAULT_THRESHOLD,
                 ambiguity_margin=DEFAULT_AMBIGUITY_MARGIN):
        for name, value in (("threshold", threshold),
                            ("ambiguity_margin", ambiguity_margin)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} deve ser um número entre 0 e 1.")
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        path = Path(knowledge_path)
        try:
            with path.open(encoding="utf-8") as source:
                self.knowledge = json.load(source)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Não foi possível ler a base {path}: {exc}") from exc

        if not isinstance(self.knowledge, list) or not self.knowledge:
            raise ValueError("A base deve ser uma lista não vazia de orientações.")
        for index, entry in enumerate(self.knowledge, start=1):
            if not isinstance(entry, dict) or any(
                not isinstance(entry.get(field), str) or not entry[field].strip()
                for field in ("intent", "question", "answer")
            ):
                raise ValueError(
                    f"Entrada {index}: intent, question e answer devem ser textos não vazios."
                )
            if entry["intent"] not in INTENTS:
                raise ValueError(f"Entrada {index}: intenção inválida {entry['intent']!r}.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2)
        )
        try:
            self.question_vectors = self.vectorizer.fit_transform(
                [entry["question"] for entry in self.knowledge]
            )
        except ValueError as exc:
            raise ValueError("A base não contém vocabulário utilizável pelo TF-IDF.") from exc
        if (self.question_vectors.getnnz(axis=1) == 0).any():
            raise ValueError("Cada pergunta da base deve conter vocabulário utilizável.")

    def respond(self, message):
        if not isinstance(message, str):
            raise TypeError("A mensagem deve ser um texto.")
        result = {
            "intent": "fallback", "answer": FALLBACK_ANSWER,
            "similarity": 0.0, "matched_question": None,
        }
        if not message.strip():
            return result
        vector = self.vectorizer.transform([message])
        if vector.nnz == 0:
            return result
        scores = cosine_similarity(vector, self.question_vectors).ravel()
        best_index = int(scores.argmax())
        best = self.knowledge[best_index]
        score = float(scores[best_index])
        result.update(similarity=score, matched_question=best["question"])
        if score < self.threshold:
            return result

        competing_score = max(
            (float(scores[i]) for i, entry in enumerate(self.knowledge)
             if entry["intent"] != best["intent"]), default=0.0
        )
        if (competing_score > 0 and competing_score >= self.threshold
                and score - competing_score <= self.ambiguity_margin):
            result["answer"] = AMBIGUOUS_ANSWER
            return result
        result.update(intent=best["intent"], answer=best["answer"])
        return result
