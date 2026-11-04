"""Testes de integração do terminal com uma base temporária sintética."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main as terminal


BASE = [
    {"intent": "acesso", "question": "Esqueci minha senha", "answer": "Recupere a senha fictícia."},
    {"intent": "hardware", "question": "Monitor apagado", "answer": "Confira o cabo externo."},
]


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "knowledge.json"
        self.write_base(BASE)

    def write_base(self, entries):
        self.path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    def run_terminal(self, messages, debug=False):
        output = io.StringIO()
        with patch.object(terminal, "KNOWLEDGE_PATH", self.path), \
                patch("sys.argv", ["main.py"] + (["--debug"] if debug else [])), \
                patch("builtins.input", side_effect=messages), \
                patch("sys.stdout", output):
            code = terminal.main()
        return code, output.getvalue()

    def test_terminal_conversation(self):
        code, output = self.run_terminal(["ajuda", "", "Esqueci minha senha", " SAIR "])
        self.assertEqual(code, 0)
        self.assertIn(BASE[0]["answer"], output)
        self.assertIn("Não reconheci", output)
        self.assertNotIn("[debug]", output)

    def test_terminal_debug(self):
        code, output = self.run_terminal(["Monitor apagado", "sair"], debug=True)
        self.assertEqual(code, 0)
        self.assertIn("[debug] intenção=hardware", output)

    def test_terminal_interruptions(self):
        for interruption in (EOFError, KeyboardInterrupt):
            with self.subTest(interruption=interruption):
                code, output = self.run_terminal([interruption])
                self.assertEqual(code, 0)
                self.assertIn("encerrado", output)

    def test_terminal_invalid_base(self):
        self.write_base([])
        code, output = self.run_terminal([])
        self.assertEqual(code, 1)
        self.assertIn("Não foi possível iniciar", output)


if __name__ == "__main__":
    unittest.main()
