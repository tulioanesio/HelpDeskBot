import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tests.evaluate import evaluate, export_results, load_cases, summarize


class EvaluationTests(unittest.TestCase):
    def row(self, **criteria):
        engine = Mock()
        engine.respond.return_value = {
            'intent': 'fallback', 'answer': 'Descreva um problema por vez',
            'similarity': 0.5, 'matched_question': 'Pergunta de referência',
        }
        return evaluate([{'message': 'dois problemas', 'expected_intent': 'fallback',
                          **criteria}], engine)[0]

    def test_absent_or_empty_criteria_are_not_approved(self):
        for criteria in ({}, {'expected_answer_contains': []}):
            with self.subTest(criteria=criteria):
                row = self.row(**criteria)
                self.assertIsNone(row['answer_ok'])
                self.assertIsNone(row['pass'])
                self.assertEqual(row['verdict'], 'NOT_EVALUATED')

    def test_human_review_does_not_automatically_pass(self):
        row = self.row(expected_answer_contains=['Descreva'], requires_human_review=True)
        self.assertIsNone(row['answer_ok'])

    def test_intent_and_answer_are_independent(self):
        row = self.row(expected_answer_contains=['recuperação de senha'])
        self.assertTrue(row['intent_ok'])
        self.assertFalse(row['answer_ok'])
        self.assertEqual(row['verdict'], 'FAIL')

    def test_forbidden_phrase_rejects_specific_wrong_guidance(self):
        row = self.row(expected_answer_contains=['Descreva'],
                       expected_answer_not_contains=['um problema'])
        self.assertFalse(row['answer_ok'])

    def test_missing_reason_never_confirms_ambiguity(self):
        row = self.row(expected_answer_contains=['Descreva'],
                       expected_fallback_reason='ambiguous')
        self.assertIsNone(row['fallback_reason'])
        self.assertIsNone(row['fallback_reason_ok'])
        self.assertEqual(row['verdict'], 'NOT_EVALUATED')

    def test_explicit_reasons_are_compared_separately(self):
        for reason, expected in [('low_similarity', False), ('ambiguous', True)]:
            with self.subTest(reason=reason):
                engine = Mock()
                engine.respond.return_value = {
                    'intent': 'fallback', 'answer': 'Descreva', 'similarity': 0.5,
                    'matched_question': None, 'fallback_reason': reason,
                }
                row = evaluate([{'message': 'teste', 'expected_intent': 'fallback',
                                 'expected_answer_contains': ['Descreva'],
                                 'expected_fallback_reason': 'ambiguous'}], engine)[0]
                self.assertIs(row['fallback_reason_ok'], expected)
                self.assertIs(row['pass'], expected)

    def test_metrics_exclude_unreviewed_answers_from_denominator(self):
        metrics = summarize([self.row(), self.row(expected_answer_contains=['Descreva']),
                             self.row(expected_answer_contains=['ausente'])])
        self.assertEqual(metrics['answer_evaluated'], 2)
        self.assertEqual(metrics['answer_not_evaluated'], 1)
        self.assertEqual(metrics['answer_criterion_rate'], 0.5)
        self.assertEqual(metrics['fallback_reasons'], {'unavailable': 3})
        self.assertIsNone(summarize([])['answer_criterion_rate'])

    def test_rejects_malformed_criteria_and_intents(self):
        invalid = [
            {'expected_answer_contains': ['']},
            {'expected_answer_contains': [2]},
            {'expected_answer_not_contains': 'texto'},
            {'expected_intent': []},
            {'expected_intent': 'inexistente'},
            {'expected_fallback_reason': []},
            {'expected_fallback_reason': 'inventado'},
            {'requires_human_review': 'sim'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.json'
            for change in invalid:
                with self.subTest(change=change):
                    path.write_text(json.dumps([{'message': '', 'expected_intent': 'fallback', **change}]))
                    with self.assertRaises(ValueError):
                        load_cases(path)

    def test_export_preserves_unreviewed_results_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested/results.json'
            engine = Mock(threshold=0.35, ambiguity_margin=0.05)
            export_results(path, [self.row()], engine)
            saved = json.loads(path.read_text())
            self.assertIsNone(saved['rows'][0]['answer_ok'])
            self.assertIsNone(saved['rows'][0]['fallback_reason'])
            self.assertEqual(saved['metrics']['answer_not_evaluated'], 1)
            self.assertEqual(saved['metadata']['threshold'], 0.35)
            self.assertEqual(len(saved['metadata']['sha256']['cases']), 64)

    def test_exploratory_cases_include_five_ambiguity_scenarios(self):
        cases = load_cases()
        self.assertTrue(all(c['evaluation_set'] == 'exploratory' for c in cases))
        ambiguous = [c for c in cases if c['category'] == 'ambiguity']
        self.assertGreaterEqual(len(ambiguous), 5)
        self.assertTrue(any(c['expected_intent'] == 'rede' for c in ambiguous))
        self.assertEqual(len({c['id'] for c in cases}), len(cases))


if __name__ == '__main__':
    unittest.main()
