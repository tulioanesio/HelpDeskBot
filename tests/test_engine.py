"""Testes técnicos com dados sintéticos; não são a avaliação acadêmica."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chatbot.engine import AMBIGUOUS_ANSWER, ChatbotEngine


BASE = [
    {"intent": "acesso", "question": "Esqueci minha senha", "answer": "Recupere a senha fictícia."},
    {"intent": "acesso", "question": "Conta bloqueada", "answer": "Solicite desbloqueio simulado."},
    {"intent": "rede", "question": "Conexão instável", "answer": "Verifique a conexão."},
    {"intent": "hardware", "question": "Monitor apagado", "answer": "Confira o cabo externo."},
    {"intent": "impressora", "question": "Papel preso", "answer": "Solicite suporte simulado."},
]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "knowledge.json"
        self.write_base(BASE)

    def write_base(self, entries):
        self.path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    def test_recovers_each_specific_answer(self):
        engine = ChatbotEngine(self.path)
        for entry in BASE:
            with self.subTest(question=entry["question"]):
                result = engine.respond(entry["question"])
                self.assertEqual(result["intent"], entry["intent"])
                self.assertEqual(result["answer"], entry["answer"])
                self.assertEqual(result["matched_question"], entry["question"])
                self.assertAlmostEqual(result["similarity"], 1.0)

    def test_normalization(self):
        result = ChatbotEngine(self.path).respond("CONEXAO INSTAVEL!")
        self.assertEqual(result["intent"], "rede")

    def test_empty_and_unknown_even_with_zero_threshold(self):
        engine = ChatbotEngine(self.path, threshold=0)
        for message in ("", "  ", "!!!", "xyzqv"):
            with self.subTest(message=message):
                result = engine.respond(message)
                self.assertEqual(result["intent"], "fallback")
                self.assertEqual(result["similarity"], 0.0)
                self.assertIsNone(result["matched_question"])

    def test_low_similarity_preserves_metadata(self):
        result = ChatbotEngine(self.path, threshold=0.99).respond("senha")
        self.assertEqual(result["intent"], "fallback")
        self.assertGreater(result["similarity"], 0)
        self.assertLess(result["similarity"], 0.99)
        self.assertEqual(result["matched_question"], BASE[0]["question"])

    def test_ambiguity_between_intents(self):
        result = ChatbotEngine(self.path).respond("Monitor apagado papel preso")
        self.assertEqual(result["intent"], "fallback")
        self.assertEqual(result["answer"], AMBIGUOUS_ANSWER)

    def test_same_intent_tie_is_not_ambiguous(self):
        self.write_base([BASE[0], dict(BASE[0])])
        self.assertEqual(ChatbotEngine(self.path).respond("Esqueci minha senha")["intent"], "acesso")

    def test_different_intent_checked_beyond_second_question(self):
        self.write_base([BASE[2], dict(BASE[2]), {**BASE[2], "intent": "hardware"}])
        result = ChatbotEngine(self.path).respond(BASE[2]["question"])
        self.assertEqual(result["answer"], AMBIGUOUS_ANSWER)

    def test_does_not_refit_during_conversation(self):
        engine = ChatbotEngine(self.path)
        with patch.object(engine.vectorizer, "fit_transform", side_effect=AssertionError("refit")):
            engine.respond("senha")
            engine.respond("monitor")

    def test_invalid_bases(self):
        for entries in ([], {}, [None], [{}], [{**BASE[0], "answer": " "}],
                        [{**BASE[0], "intent": "rh"}],
                        [{**BASE[0], "question": "!!!"}],
                        [BASE[0], {**BASE[1], "question": "!!!"}]):
            with self.subTest(entries=entries):
                self.write_base(entries)
                with self.assertRaises(ValueError):
                    ChatbotEngine(self.path)

    def test_missing_and_invalid_json(self):
        with self.assertRaises(ValueError):
            ChatbotEngine(self.path.parent / "missing.json")
        self.path.write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            ChatbotEngine(self.path)

    def test_invalid_parameters_and_message(self):
        for parameter in ("threshold", "ambiguity_margin"):
            for value in (-1, 2, float("nan"), float("inf"), None, "0.35", True):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        ChatbotEngine(self.path, **{parameter: value})
        with self.assertRaises(TypeError):
            ChatbotEngine(self.path).respond(None)


if __name__ == "__main__":
    unittest.main()
