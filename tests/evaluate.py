"""Avaliação exploratória: critérios textuais não substituem revisão humana."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import sklearn

from chatbot.engine import ChatbotEngine, INTENTS


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = Path(__file__).with_name("cases.json")
KNOWLEDGE_PATH = ROOT / "data" / "knowledge.json"
REASONS = {"ambiguous", "low_similarity", "empty_message"}


def load_cases(path=CASES_PATH):
    path = Path(path)
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Não foi possível ler os casos {path}: {exc}") from exc
    if not isinstance(cases, list) or not cases:
        raise ValueError("A lista de casos deve conter pelo menos um caso.")
    for number, case in enumerate(cases, start=1):
        if (not isinstance(case, dict)
                or not isinstance(case.get("message"), str)
                or not isinstance(case.get("expected_intent"), str)
                or case.get("expected_intent") not in INTENTS | {"fallback"}):
            raise ValueError(f"Caso {number}: mensagem ou intenção inválida.")
        for key in ("expected_answer_contains", "expected_answer_not_contains"):
            phrases = case.get(key, [])
            if not isinstance(phrases, list) or any(
                    not isinstance(p, str) or not p.strip() for p in phrases):
                raise ValueError(f"Caso {number}: {key} exige lista de textos não vazios.")
        if ("requires_human_review" in case
                and not isinstance(case["requires_human_review"], bool)):
            raise ValueError(f"Caso {number}: requires_human_review deve ser booleano.")
        if "expected_fallback_reason" in case and (
                case["expected_intent"] != "fallback"
                or not isinstance(case["expected_fallback_reason"], str)
                or case["expected_fallback_reason"] not in REASONS):
            raise ValueError(f"Caso {number}: motivo esperado de fallback inválido.")
    return cases


def evaluate(cases, engine):
    rows = []
    for case in cases:
        result = engine.respond(case["message"])
        intent_ok = result["intent"] == case["expected_intent"]
        required = case.get("expected_answer_contains", [])
        forbidden = case.get("expected_answer_not_contains", [])
        answer = result["answer"].casefold()
        answer_ok = None
        if (required or forbidden) and not case.get("requires_human_review", False):
            answer_ok = (all(p.casefold() in answer for p in required)
                         and all(p.casefold() not in answer for p in forbidden))
        fallback = result["intent"] == "fallback"
        # O motivo deve vir do motor; não inferir pela resposta ou pela pontuação.
        reason = result.get("fallback_reason") if fallback else None
        reason_ok = None
        if "expected_fallback_reason" in case:
            if not fallback:
                reason_ok = False
            elif reason is not None:
                reason_ok = reason == case["expected_fallback_reason"]
        checks = [intent_ok, answer_ok]
        if "expected_fallback_reason" in case:
            checks.append(reason_ok)
        verdict = ("FAIL" if any(c is False for c in checks) else
                   "NOT_EVALUATED" if any(c is None for c in checks) else "PASS")
        rows.append({
            "case": case, "result": result, "intent_ok": intent_ok,
            "answer_ok": answer_ok, "fallback": fallback,
            "fallback_reason": reason, "fallback_reason_ok": reason_ok,
            "verdict": verdict,
            "pass": True if verdict == "PASS" else False if verdict == "FAIL" else None,
        })
    return rows


def summarize(rows):
    total = len(rows)
    answer_evaluated = sum(r["answer_ok"] is not None for r in rows)
    answer_correct = sum(r["answer_ok"] is True for r in rows)
    intents_correct = sum(r["intent_ok"] for r in rows)
    fallbacks = sum(r["fallback"] for r in rows)
    by_intent = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        group = by_intent[row["case"]["expected_intent"]]
        group["total"] += 1
        group["correct"] += row["intent_ok"]
    return {
        "total": total, "intent_correct": intents_correct,
        "intent_accuracy": intents_correct / total if total else None,
        "answer_evaluated": answer_evaluated, "answer_correct": answer_correct,
        "answer_not_evaluated": total - answer_evaluated,
        "answer_criterion_rate": answer_correct / answer_evaluated if answer_evaluated else None,
        "fallbacks": fallbacks, "fallback_rate": fallbacks / total if total else None,
        "fallback_reasons": dict(Counter(r["fallback_reason"] or "unavailable"
                                         for r in rows if r["fallback"])),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "by_intent": dict(by_intent),
        "confusion": [
            {"expected": expected, "obtained": obtained, "count": count}
            for (expected, obtained), count in sorted(Counter(
                (r["case"]["expected_intent"], r["result"]["intent"]) for r in rows
            ).items())
        ],
    }


def print_report(rows):
    for number, row in enumerate(rows, start=1):
        case, result = row["case"], row["result"]
        adequacy = {True: "OK", False: "FALHOU", None: "NÃO AVALIADA"}[row["answer_ok"]]
        print(f"[{number:02d}] {row['verdict']} | {case['message']!r}")
        print(f"  Intenção esperada/obtida: {case['expected_intent']} / {result['intent']}")
        print(f"  Critérios da resposta: {adequacy}; fallback: {row['fallback']}; "
              f"motivo: {row['fallback_reason'] or ('indisponível' if row['fallback'] else 'não se aplica')}")
        print(f"  Similaridade: {result['similarity']:.3f}; correspondência: {result['matched_question']!r}")
        print(f"  Resposta: {result['answer']}")
    print("\nMétricas observadas (adequação automática por critérios textuais):")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


def export_results(path, rows, engine, cases_path=CASES_PATH):
    sources = {"cases": Path(cases_path), "knowledge": KNOWLEDGE_PATH,
               "engine": ROOT / "chatbot/engine.py", "evaluator": Path(__file__)}
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "threshold": engine.threshold, "ambiguity_margin": engine.ambiguity_margin,
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "sha256": {name: hashlib.sha256(source.read_bytes()).hexdigest()
                       for name, source in sources.items()},
            "evaluation_set": "exploratory",
            "limitations": "Critérios textuais não substituem revisão humana. Motivos ausentes não são inferidos.",
        },
        "metrics": summarize(rows), "rows": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "results/evaluation.json")
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.cases)
        engine = ChatbotEngine(KNOWLEDGE_PATH)
        rows = evaluate(cases, engine)
        export_results(args.output, rows, engine, args.cases)
    except (OSError, ValueError) as exc:
        print(f"Avaliação não executada: {exc}")
        return 1
    print(f"Avaliação exploratória de {len(cases)} casos; "
          f"limiar={engine.threshold:.2f}, margem={engine.ambiguity_margin:.2f}.\n")
    print_report(rows)
    print(f"\nResultados exportados: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
