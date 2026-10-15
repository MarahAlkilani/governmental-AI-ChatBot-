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

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Set Up Ollama

Install Ollama from:

https://ollama.com

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

The model must already be installed or provided by the hackathon organizers.

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

The system uses the Arabic government-service documents supplied by the hackathon organizers.

For the practice participant pack, the expected directory structure is:

```text
hackathon_participant_pack/
└── participant_pack/
    ├── questions.json
    └── data/
        └── *.txt
```

Keep the original Arabic document content intact.

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

The CLI generates a UTF-8 `answers.json` file beside the input JSON file.

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

The project includes an optional Streamlit interface for interactive demonstrations.

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

Run the automated tests:

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
