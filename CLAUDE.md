# Kashaf (كشّاف) - AI Future 26 hackathon

Arabic RAG chatbot over 12 Jordanian civil-status service files (passport / family book / smart ID).

## Judged entry point
`python main.py questions.json` -> writes `answers.json` next to it. Every message needs
`answer` and `contexts` (the data chunks actually used). Hidden questions: dialect, age-based fees,
facts in notes, comparisons, follow-ups, unclear, out-of-scope. Internet is OFF during judging.

## Layout
- `rag_engine.py` - loader, field-based chunking (1200/200, separators `["\n\n","\n",".", " "]`), question understanding (document x situation x intent, dialect, follow-up memory), hybrid retrieval (bge-m3 + lexical), llama3 generation, exact refusal sentence.
- `main.py` - judged entry point. `evaluate.py` - rich auto-grader + answers.json audit.
- `app.py` - Streamlit UI (never hide the sidebar toggle; timeline via st.markdown unsafe_allow_html, never components.v1.html).
- `Makefile` - `make install | run | eval | main | pull`.

## Rules
- Answers only from `data/`; refusal is exactly: عذراً، المعلومات غير متوفرة في الوثائق.
- Always open files with encoding="utf-8". Never let one message crash the run.
- Env overrides: LLM_MODEL (default llama3), EMBED_MODEL (bge-m3), OLLAMA_HOST, NUM_CTX.
- Models must be pulled beforehand: `make pull`.
