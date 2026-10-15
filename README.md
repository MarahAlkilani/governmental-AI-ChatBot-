# 🇯🇴 Arabic Government Services RAG Chatbot

**Offline Arabic Government Services Assistant using Retrieval-Augmented Generation (RAG), BM25, Ollama, and Streamlit.**

This project was developed for an AI Hackathon to help users find accurate information about government services in Arabic.

The system retrieves relevant information from organizer-provided government-service documents and uses a local language model to generate Arabic answers grounded in those documents.

The chatbot runs locally without cloud APIs during inference.

---

## ✨ Features

- **Arabic Language Support:** Arabic text normalization and tokenization.
- **Retrieval-Augmented Generation (RAG):** Answers questions using relevant document passages.
- **BM25 Search:** Retrieves relevant government-service information.
- **Local Ollama Model:** Uses `command-r7b-arabic:7b`.
- **Grounded Answers:** Includes exact source excerpts in the CLI output.
- **Conversation History:** Maintains history separately within each chat.
- **Safe Fallbacks:** Handles missing evidence and model failures without inventing information.
- **Streamlit Frontend:** Optional interactive chatbot interface.
- **Automated Evaluation:** Processes JSON questions and generates JSON answers.
- **Offline Runtime:** No cloud APIs or model downloads required during execution.

---

## 🧠 System Architecture

```text
User Question
     |
     v
Arabic Text Normalization
     |
     v
BM25 Document Retrieval
     |
     v
Relevant Document Passages
     |
     v
Ollama Local Language Model
(command-r7b-arabic:7b)
     |
     v
Grounded Arabic Answer
     |
     v
Answer + Exact Source Contexts
```

The retrieval component finds relevant passages from the provided documents.

The generation component uses the local Ollama model to answer the question based on retrieved evidence.

For CLI evaluation, the model selects source line ranges, and Python copies the selected lines verbatim into the output.

Semantic relevance and answer completeness still depend on retrieval quality and model judgment.

---

## 🛠️ Technologies

| Component | Technology |
|---|---|
| Language | Python |
| Retrieval | BM25 (`rank-bm25`) |
| Arabic Processing | Text normalization and tokenization |
| LLM | `command-r7b-arabic:7b` |
| Local Model Runtime | Ollama |
| Frontend | Streamlit |
| Input / Output | JSON |
| Testing | Pytest |
| Environment | Local / Offline |

---

## 📁 Project Structure

```text
governmental-AI-ChatBot-/
│
├── app.py                 # Streamlit frontend
├── main.py                # Main CLI entry point
├── retrieval.py           # BM25 retrieval
├── generation.py          # Ollama answer generation
├── arabic_text.py         # Arabic text processing
├── ingest.py              # Document ingestion
├── config.py              # Configuration
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
│
├── docs/
│   ├── architecture.md
│   └── contracts.md
│
└── tests/                 # Automated tests
```

Organizer-provided documents and practice questions may be distributed separately from the public repository.

---

## 🚀 Installation

Complete installation while internet access is available. Run setup and evaluation commands from the project root; reuse an existing configured `.venv`.

### 1. Clone the Repository

```bash
git clone https://github.com/MarahAlkilani/governmental-AI-ChatBot-.git
cd governmental-AI-ChatBot-
```

### 2. Create a Virtual Environment

**Linux / WSL Ubuntu:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Ubuntu reports that venv is unavailable, install its `python3-venv` package first.

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Set Up Ollama

Install Ollama from:

[Official Linux instructions](https://docs.ollama.com/linux) or [Ollama downloads](https://ollama.com). The local server must be available at `http://localhost:11434`.

Ensure the organizer-provided model is available locally:

```bash
ollama list
```

Required model:

```text
command-r7b-arabic:7b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

The model must already be installed or provided by the hackathon organizers. Verify the exact tag in `ollama list`; do not substitute another model. Export the model setting in the same terminal used to run the CLI or demo.

### 5. Configure the Model

**Linux / WSL Ubuntu:**

```bash
export OLLAMA_MODEL=command-r7b-arabic:7b
```

**Windows PowerShell:**

```powershell
$env:OLLAMA_MODEL="command-r7b-arabic:7b"
```

---

## 📚 Government Service Documents

Place the 12 original service TXT files in `data/` beside `main.py`. Alternatively, keep the supplied package layout:

```text
hackathon_participant_pack/participant_pack/data/*.txt
hackathon_participant_pack/participant_pack/questions.json
```

The CLI selects one complete, readable UTF-8 corpus of the 12 official service TXT filenames in this order:

1. `--documents path/to/data`, or `GOVERNMENT_RAG_DATA_DIR` when the flag is omitted, if valid.
2. `data/` beside the supplied questions file.
3. The project's `docs/` directory, if it contains the official TXT corpus.

Invalid candidates are skipped; if none is valid, the CLI fails with the paths checked. It never combines datasets or ingests unrelated TXT or architecture files. User-supplied paths are relative to the current working directory; repository defaults are relative to the code directory. Preserve filenames and original text. Organizer files are excluded from Git and must be supplied separately on a fresh checkout.

---

## 💻 Run the CLI

The CLI is the primary interface for automated hackathon evaluation.

Activate the virtual environment, configure Ollama, and run:

```bash
python main.py questions.json
```

For the supplied practice questions:

```bash
python main.py hackathon_participant_pack/participant_pack/questions.json
```

The CLI accepts any explicit questions path, including `participant_pack/questions.json`, and reports resolved input, corpus, sibling `answers_example.json` (if present), and output paths. It never modifies questions or reference examples.

The CLI writes UTF-8 `answers.json` **beside the input file**, replacing any prior output there. Root `questions.json` produces root `answers.json`; the nested practice command produces `hackathon_participant_pack/participant_pack/answers.json`. A root copy of the practice output is not refreshed by the nested command.

Input is an object containing `chats`, each with `chat_id` and `messages`; each message contains `message_id` and a string `question`. Output preserves these fields and adds an Arabic string `answer` and a list of exact source-text strings `contexts` to every message. Abstentions and local model failures have empty contexts. See `docs/contracts.md` for the full interface contract.

After dependencies and the model are installed, execution is fully offline: requests go only to local Ollama, with no cloud APIs or runtime downloads. History stays within each chat. The model selects source lines, which Python copies verbatim. Relevance and completeness still depend on retrieval and model judgment; service-free follow-ups can abstain. Local model errors appear on stderr. `OLLAMA_TIMEOUT_SECONDS` optionally overrides the default 45-second timeout.


### Output Format

```json
{
  "chats": [
    {
      "chat_id": "chat-01",
      "messages": [
        {
          "message_id": "chat-01-1",
          "question": "ما هي الوثائق المطلوبة للحصول على الخدمة؟",
          "answer": "تُحدَّد الوثائق المطلوبة وفقًا للمعلومات المتاحة في دليل الخدمة.",
          "contexts": [
            "مقطع نصي أصلي من وثيقة الخدمة"
          ]
        }
      ]
    }
  ]
}
```

This example illustrates the output structure and is not an actual government-service answer.

Each message receives an Arabic answer and a `contexts` array containing selected source excerpts.

When supporting evidence is unavailable or the local model fails, the system produces an Arabic fallback answer. In such cases, `contexts` may be empty.

---

## 🖥️ Streamlit Frontend

The project includes `app.py`, an optional Streamlit demo using the existing retrieval and generation APIs. Activate the same environment and model setting first. To select its corpus explicitly, set `GOVERNMENT_RAG_DATA_DIR` to the official TXT directory (for example, `hackathon_participant_pack/participant_pack/data`). The demo displays retrieved evidence; the CLI performs the official used-evidence mapping and writes the evaluation JSON.

Run:

```bash
python -m streamlit run app.py
```

Then open:

http://localhost:8501

The frontend provides an interactive way to demonstrate the chatbot.

**Note:** The Streamlit interface is optional. The CLI remains the primary workflow for automated evaluation.

---

## 🧪 Testing

In the activated environment, install `pytest` beforehand if needed (it is not listed in runtime requirements), then run:

```bash
python -m pytest -q
```

Tests validate implementation behavior and expected output structures.

Passing automated tests does not guarantee the semantic correctness of every generated answer.

---

## 🔒 Offline Execution

The system is designed to operate locally using:

- Government-service documents stored on the machine.
- Python dependencies installed beforehand.
- The organizer-provided Ollama model.

No cloud APIs or model downloads are required during runtime.

Initial installation of dependencies and Ollama may require internet access.

---

## 🎯 Project Objectives

1. Make Arabic government-service information easier to access.
2. Retrieve relevant information from provided documents.
3. Generate reliable answers supported by textual evidence.
4. Reduce unsupported or fabricated answers.
5. Support offline execution and reproducible evaluation.
6. Provide a simple interactive demonstration through Streamlit.

---

## 👩‍💻 Team Members

**Marah Alkilani**
- Backend development
- Ollama integration
- Answer generation
- CLI implementation

**Sadeen Nababateh**
- Testing
- Evaluation
- Integration

**Bayan Marashdeh**
- Document processing
- Arabic text normalization
- BM25 retrieval
- Streamlit frontend

---

## 📌 Notes for Hackathon Evaluators

| Item | Details |
|---|---|
| Primary entry point | `main.py` |
| Optional interface | `app.py` |
| Required model | `command-r7b-arabic:7b` |
| Input file | `questions.json` |
| Output file | `answers.json` |
| Evidence format | Exact document excerpts in `contexts` |
| Execution | Local Python + Ollama |
| Cloud APIs | Not required |

The system prioritizes **retrieval relevance, grounded Arabic answers, offline execution, and safe handling of missing information**.

---

**Developed for the AI Hackathon 🇯🇴**

**Team: Marah Alkilani · Sadeen Nababateh · Bayan Marashdeh**
