# Arabic Government Services RAG

Offline Arabic document retrieval and answer generation using the provided local Ollama model.

```bash
source .venv/bin/activate
export OLLAMA_MODEL=command-r7b-arabic:7b
python main.py questions.json
```

Install the dependencies in requirements.txt beforehand. No model downloads or cloud APIs are used at runtime.
The CLI writes UTF-8 answers.json beside the input. For the supplied practice input:

```bash
python main.py hackathon_participant_pack/participant_pack/questions.json
```

Each message receives an Arabic answer and exact source excerpts. Missing evidence or local model failures produce Arabic fallback answers. History remains within each chat. The model selects source line ranges; Python copies those lines verbatim, avoiding formatting-related rejections. Semantic relevance and completeness still depend on retrieval and model judgment.

Run tests with `python -m pytest -q`.
