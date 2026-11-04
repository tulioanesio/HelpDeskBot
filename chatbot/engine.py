"""Espaço reservado para o motor de PLN, sem entrada/saída de terminal.

Implementação futura conforme context.md:
- Carregar e validar a base com intent, question e answer.
- Ajustar o TF-IDF uma vez e comparar mensagens por similaridade de cosseno.
- Retornar a resposta da pergunta mais próxima ou fallback.
- Expor intent, answer, similarity e matched_question no resultado.

Interface planejada: ChatbotEngine(knowledge_path, threshold=0.35).respond(message).
O limiar inicial é uma hipótese a calibrar com testes independentes.
"""
