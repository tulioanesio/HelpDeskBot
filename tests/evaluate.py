import json
from collections import Counter, defaultdict
from pathlib import Path

from chatbot.engine import ChatbotEngine


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = Path(__file__).with_name("cases.json")
KNOWLEDGE_PATH = ROOT / "data" / "knowledge.json"


def load_cases(path=CASES_PATH):
    try:
        with path.open(encoding="utf-8") as source:
            cases = json.load(source)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Não foi possível ler os casos {path}: {exc}") from exc
    if not isinstance(cases, list) or not cases:
        raise ValueError("A lista de casos deve conter pelo menos um caso.")
    for number, case in enumerate(cases, start=1):
        if (not isinstance(case, dict)
                or not isinstance(case.get("message"), str)
                or not isinstance(case.get("expected_intent"), str)):
            raise ValueError(
                f"Caso {number}: message e expected_intent devem ser textos."
            )
        if "expected_answer_contains" in case and not isinstance(
                case["expected_answer_contains"], list):
            raise ValueError(
                f"Caso {number}: expected_answer_contains deve ser uma lista."
            )
    return cases


def evaluate(cases, engine):
    rows = []
    for case in cases:
        result = engine.respond(case["message"])
        expected = case["expected_intent"]
        intent_ok = result["intent"] == expected
        required_phrases = case.get("expected_answer_contains", [])
        answer_ok = all(
            phrase.casefold() in result["answer"].casefold()
            for phrase in required_phrases
        )
        rows.append({
            "case": case,
            "result": result,
            "intent_ok": intent_ok,
            "answer_ok": answer_ok,
            "pass": intent_ok and answer_ok,
        })
    return rows


def print_report(rows):
    confusion = Counter()
    by_intent = defaultdict(lambda: [0, 0])
    fallbacks = 0
    answer_passes = 0
    full_passes = 0

    for number, row in enumerate(rows, start=1):
        case, result = row["case"], row["result"]
        expected = case["expected_intent"]
        obtained = result["intent"]
        confusion[(expected, obtained)] += 1
        by_intent[expected][1] += 1
        by_intent[expected][0] += row["intent_ok"]
        fallbacks += obtained == "fallback"
        answer_passes += row["answer_ok"]
        full_passes += row["pass"]
        expected_phrases = case.get("expected_answer_contains", [])
        print(f"[{number:02d}] {'PASS' if row['pass'] else 'FAIL'}")
        print(f"  Entrada: {case['message']!r}")
        print(f"  Intenção esperada/obtida: {expected} / {obtained}")
        if expected_phrases:
            print(f"  Critério da resposta: conter {expected_phrases!r} — "
                  f"{'OK' if row['answer_ok'] else 'FALHOU'}")
        print(f"  Similaridade: {result['similarity']:.3f} | "
              f"Correspondência: {result['matched_question']!r}")
        print(f"  Resposta: {result['answer']}")

    total = len(rows)
    correct_intents = sum(row["intent_ok"] for row in rows)
    print("\nMétricas observadas")
    print(f"Acurácia de intenção: {correct_intents}/{total} "
          f"({correct_intents / total:.1%})")
    print(f"Casos com critério de resposta atendido: {answer_passes}/{total} "
          f"({answer_passes / total:.1%})")
    print(f"Casos totalmente aprovados: {full_passes}/{total} "
          f"({full_passes / total:.1%})")
    print(f"Taxa de fallback: {fallbacks}/{total} ({fallbacks / total:.1%})")
    print("Acurácia por intenção:")
    for intent, (correct, count) in sorted(by_intent.items()):
        print(f"  {intent}: {correct}/{count} ({correct / count:.1%})")
    print("Matriz de confusão (esperada → obtida):")
    for (expected, obtained), count in sorted(confusion.items()):
        print(f"  {expected} → {obtained}: {count}")
    failures = [row for row in rows if not row["pass"]]
    if failures:
        print("\nCasos para análise de falha (inspecionar causa antes de concluir):")
        for row in failures:
            print(f"  - {row['case']['message']!r}: esperado "
                  f"{row['case']['expected_intent']}, obtido "
                  f"{row['result']['intent']}; similaridade "
                  f"{row['result']['similarity']:.3f}.")
    else:
        print("\nNenhum erro observado nesta rodada; não há falhas para analisar.")


def main():
    try:
        cases = load_cases()
        engine = ChatbotEngine(KNOWLEDGE_PATH)
    except ValueError as exc:
        print(f"Avaliação não executada: {exc}")
        return 1
    print(f"Avaliação de {len(cases)} casos independentes; "
          f"limiar={engine.threshold:.2f}, margem={engine.ambiguity_margin:.2f}.\n")
    print_report(evaluate(cases, engine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
