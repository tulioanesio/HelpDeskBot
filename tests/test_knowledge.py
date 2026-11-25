from pathlib import Path
import unittest

from chatbot.engine import ChatbotEngine
from tests.evaluate import evaluate, load_cases


class KnowledgeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ChatbotEngine(Path(__file__).resolve().parents[1] / 'data/knowledge.json')
        cls.cases = {case['id']: case for case in load_cases()}

    def assert_case_passes(self, case_id):
        row = evaluate([self.cases[case_id]], self.engine)[0]
        self.assertTrue(row['intent_ok'], row['result'])
        self.assertTrue(row['answer_ok'], row['result'])
        self.assertIsNone(row['result']['fallback_reason'])

    def test_forgotten_password_and_blocked_account_receive_different_guidance(self):
        for case_id in ('exp-01', 'exp-06'):
            with self.subTest(case_id=case_id):
                self.assert_case_passes(case_id)

    def test_authenticator_loss_receives_second_factor_guidance(self):
        self.assert_case_passes('exp-05')

    def test_overheating_does_not_receive_network_guidance(self):
        self.assert_case_passes('exp-18')

    def test_empty_tray_and_jammed_page_receive_specific_guidance(self):
        for case_id in ('exp-21', 'exp-22'):
            with self.subTest(case_id=case_id):
                self.assert_case_passes(case_id)


if __name__ == '__main__':
    unittest.main()
