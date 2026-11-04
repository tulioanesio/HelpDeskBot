import argparse
from pathlib import Path

from chatbot.engine import ChatbotEngine


KNOWLEDGE_PATH = Path(__file__).resolve().parent / "data" / "knowledge.json"
HELP = (
    "Posso orientar sobre acesso, rede, hardware e impressoras em uma empresa "
    "fictícia. Não realizo ações reais nem diagnósticos definitivos. "
    "RH, financeiro, notícias, reparos perigosos e acesso não autorizado "
    "estão fora do escopo. Não informe senhas ou dados pessoais. "
    "Digite ajuda para rever estas instruções ou sair para encerrar."
)


def main():
    parser = argparse.ArgumentParser(description="HelpDeskBot — suporte de TI fictício")
    parser.add_argument("--debug", action="store_true", help="Exibe metadados da busca")
    args = parser.parse_args()
    try:
        engine = ChatbotEngine(KNOWLEDGE_PATH)
    except ValueError as exc:
        print(f"Não foi possível iniciar o HelpDeskBot: {exc}")
        print("Revise data/knowledge.json conforme o contrato descrito em context.md.")
        return 1

    print("HelpDeskBot — orientação inicial de suporte técnico fictício")
    print(HELP)
    try:
        while True:
            message = input("Você: ")
            command = message.strip().casefold()
            if command == "sair":
                break
            if command == "ajuda":
                print(HELP)
                continue
            result = engine.respond(message)
            print(f"HelpDeskBot: {result['answer']}")
            if args.debug:
                print(
                    f"[debug] intenção={result['intent']} | "
                    f"similaridade={result['similarity']:.3f} | "
                    f"pergunta={result['matched_question']!r}"
                )
    except (KeyboardInterrupt, EOFError):
        print()
    print("HelpDeskBot encerrado. Até mais!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
