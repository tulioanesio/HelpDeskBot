# HelpDeskBot

Chatbot de orientação inicial para suporte de TI, desenvolvido em Python para a avaliação A3 da disciplina de Inteligência Artificial.

O HelpDeskBot simula o atendimento a colaboradores de uma empresa fictícia. Ele recebe dúvidas em linguagem natural pelo terminal, procura uma pergunta semelhante em uma base local e apresenta uma orientação previamente revisada. Quando não encontra uma correspondência suficientemente confiável, solicita reformulação ou recomenda procurar o suporte humano simulado.

> **Protótipo acadêmico:** não acessa serviços de empresas, não abre chamados, não altera senhas ou permissões e não executa reparos. Não digite senhas, códigos de verificação ou dados pessoais.

## Funcionalidades

- Conversa interativa no terminal, com comandos `ajuda` e `sair`.
- Reconhecimento por **TF-IDF + similaridade de cosseno** (`scikit-learn`), sem LLM, API externa ou banco de dados.
- Busca por perguntas semelhantes em `data/knowledge.json` e recuperação da resposta cadastrada.
- **Fallback** para mensagens vazias, desconhecidas ou com pontuação insuficiente.
- Solicitação de esclarecimento quando duas intenções diferentes têm pontuações muito próximas.
- Modo `--debug` para observar intenção, similaridade e pergunta encontrada.
- Testes unitários do motor e da interface de terminal.

## Escopo do atendimento

| Intenção | Exemplos de assuntos |
| --- | --- |
| `acesso` | Recuperação de senha, conta bloqueada, autenticação em duas etapas e permissões de pastas |
| `rede` | Wi-Fi, acesso à internet, conexão instável, lentidão e Ethernet |
| `hardware` | Computador sem energia, monitor, teclado, mouse, lentidão e superaquecimento |
| `impressora` | Impressora offline, fila, papel, atolamento, toner e configuração de impressão |

A base atual contém **48 perguntas de referência**, distribuídas igualmente entre as quatro intenções. Os dados e procedimentos são **fictícios**, elaborados para representar situações de suporte técnico inicial.

**Fora do escopo:** RH, financeiro, notícias, suporte bancário, acesso não autorizado, diagnóstico definitivo, ações administrativas e reparos perigosos. O chatbot não garante resposta correta para toda reformulação; o desempenho deve ser avaliado com mensagens independentes da base.

## Tecnologias e requisitos

- **Python 3** (recomendado: versão compatível com a versão atual do `scikit-learn`).
- Biblioteca externa: **scikit-learn**; demais funcionalidades utilizam bibliotecas padrão do Python.
- Execução local, sem credenciais nem conexão com serviços externos após a instalação das dependências.

## Instalação

Clone o repositório e entre na pasta do projeto:

```bash
git clone https://github.com/tulioanesio/HelpDeskBot.git
cd HelpDeskBot
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative-o no **Windows (PowerShell)**:

```powershell
.venv\Scripts\Activate.ps1
```

Ou no **Linux/macOS**:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Em máquinas nas quais o comando `python` não aponta para o Python 3, use `python3` (ou `py` no Windows), conforme a instalação local.

## Como executar

Na raiz do projeto:

```bash
python main.py
```

Exemplo ilustrativo de uso:

```text
HelpDeskBot — orientação inicial de suporte técnico fictício
Você: Esqueci a senha de acesso ao computador
HelpDeskBot: No cenário fictício do HelpDeskBot, use a opção de recuperação de senha...
Você: ajuda
[O programa apresenta os temas atendidos e os limites do serviço.]
Você: sair
HelpDeskBot encerrado. Até mais!
```

Para consultar informações da decisão do algoritmo:

```bash
python main.py --debug
```

O modo de depuração mostra a **intenção**, a **similaridade numérica** e a **pergunta da base que mais se aproxima da mensagem**. A pontuação de similaridade **não é uma probabilidade de acerto**.

## Como funciona a IA/PLN

```text
Mensagem no terminal
       ↓
Validação de entrada / comandos
       ↓
Transformação em vetor TF-IDF
       ↓
Similaridade de cosseno com perguntas da base
       ↓
Seleção da pergunta de maior pontuação
       ↓
Limiar mínimo e verificação de ambiguidade
       ├── Correspondência aceita → intenção + resposta cadastrada
       └── Sem correspondência confiável → fallback / esclarecimento
```

O motor em `chatbot/engine.py` lê `data/knowledge.json`, valida suas entradas e usa `TfidfVectorizer` com normalização de caixa, remoção de acentos e unigramas/bigramas (`ngram_range=(1, 2)`). O vocabulário e os pesos TF-IDF são ajustados às perguntas de referência na inicialização. Para cada mensagem, a aplicação usa `transform()` e calcula `cosine_similarity()` para localizar a pergunta mais próxima.

## Base de conhecimento

O arquivo `data/knowledge.json` é uma lista de objetos neste formato:

```json
[
  {
    "intent": "rede",
    "question": "Meu notebook não conecta ao Wi-Fi",
    "answer": "Verifique se o Wi-Fi está ativado e se a rede correta foi selecionada."
  }
]
```

Esse trecho é apenas um **exemplo do formato**, não uma substituição da base existente. Cada item exige campos de texto não vazios `intent`, `question` e `answer`; as intenções permitidas são `acesso`, `rede`, `hardware` e `impressora`.

A base é mantida no próprio projeto e pode ser versionada no Git. Não inclua senhas, informações pessoais ou dados internos reais de empresas. A origem, a preparação e os critérios de curadoria devem ser descritos no relatório acadêmico da A3.

## Testes

Execute os testes automatizados existentes na raiz do projeto:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Os arquivos `tests/test_engine.py` e `tests/test_terminal.py` verificam comportamentos técnicos do motor e da interface. **Testes unitários não substituem a avaliação acadêmica com perguntas inéditas.**

### Avaliação de qualidade das respostas

`tests/cases.json` contém **39 casos exploratórios**, incluindo reformulações para as quatro intenções, situações fora do escopo com vocabulário de TI e cinco cenários de ambiguidade. Estes casos já foram inspecionados e não constituem uma avaliação final inédita. `tests/evaluate.py` executa os casos com o mesmo motor do terminal e imprime intenção, resposta, similaridade e veredito. Também resume acurácia de intenção, taxa de fallback, adequação por critérios textuais, resultados por intenção e confusões observadas.

Execute com:

```bash
python -m tests.evaluate
```


## Estrutura do projeto

```text
HelpDeskBot/
├── main.py                # Conversação e comandos no terminal
├── chatbot/
│   ├── __init__.py
│   └── engine.py          # Carregamento da base, TF-IDF e decisão
├── data/
│   └── knowledge.json    # Perguntas, intenções e respostas fictícias
├── tests/
│   ├── __init__.py
│   ├── test_engine.py     # Testes unitários do motor
│   ├── test_terminal.py   # Testes da interface no terminal
│   ├── cases.json         # Casos da avaliação acadêmica (pendentes)
│   └── evaluate.py       # Avaliador acadêmico (pendente)
├── requirements.txt      # Dependências
├── context.md            # Contexto do desenvolvimento
└── README.md
```

## Limitações conhecidas

- Similaridade lexical não garante que duas perguntas tenham o mesmo significado.
- O chatbot pode identificar a **intenção correta** e ainda assim devolver uma **orientação inadequada ao problema específico**.
- Mensagens fora do escopo podem atingir o limiar por compartilharem palavras com a base.
- Não há memória persistente, autenticação, abertura de chamados nem intervenção humana real.
- O fallback reduz alguns erros, mas não elimina respostas incorretas; limiar e margem ainda precisam de avaliação empírica.

## Equipe e finalidade acadêmica

Projeto desenvolvido para a **A3 de Inteligência Artificial** por:

- Tulio Anesio da Rosa
- João Pedro Farias da Silva
- Thuysa Monique Luvison da Rosa
