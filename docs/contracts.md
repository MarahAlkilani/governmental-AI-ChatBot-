# Integration Contracts (P00)

## Official JSON structure

Input questions.json is the enclosing object shown in the brief and participant file:

    {"chats": [{"chat_id": "chat-01", "messages": [
      {"message_id": "chat-01-1", "question": "..."}
    ]}]}

No formal JSON Schema or further type/validation rules are supplied.

Output answers.json preserves chats, chat_id, messages, message_id, and question exactly in place. Every message gains:

    {"answer": "text a user reads", "contexts": ["source piece used"]}

answer is text. contexts is a list of source-text strings, or [] when nothing was found. Include only source pieces used for the answer, not whole files or a summary; the brief permits cleaned or raw pieces. Every message must receive an answer. Extra output keys are permitted but ignored; this project does not rely on them.

## CLI and offline contract

    python main.py questions.json

Write answers.json beside the input/project files. Read/write Arabic with encoding="utf-8". The run may not require the internet, downloads, online services, or unavailable dependencies/models. Knowledge must come only from provided files; use the provided Ollama model at run time. Local embeddings/rerankers are allowed.

## Retrieval API

    def retrieve(question: str, top_k: int = 3) -> list[dict]:
        ...

Every result contains:

    {
        "text": str,     # exact original excerpt from a supplied TXT document
        "source": str,   # e.g. family_book_renewal.txt
        "score": float,  # numeric relevance; team convention: higher is better
    }

Return zero through top_k results. text must not be a generated summary, normalized replacement, fabricated heading, or whole document simply because it ranked. source and score are internal, not official context values. The orchestrator may resolve a follow-up with same-chat history before calling the question-only API.

## Generation API

    def generate_answer(question: str, contexts: list[dict],
                        history: list[dict] | None = None) -> str:
        ...

contexts are Retrieval API results or a selected subset. history is empty/None for the first turn, otherwise chronological earlier turns from the same chat_id only. Recommended internal turn shape:

    {"message_id": str, "question": str, "answer": str, "contexts": list[str]}

Return only user-facing answer text, never JSON, scores, filenames, or source-selection commentary. Constrain claims to supplied evidence; decline out-of-scope questions and clarify ambiguous service references.

## Evidence-to-output mapping and ownership

After generation, map supporting results actually used as:

    official_contexts = [result["text"] for result in used_results]

Keep excerpts exact and sufficient; do not replace them with an answer, synopsis, score, filename, or cleaned paraphrase. Deduplicate identical strings only without changing content. Use [] for explicit unknown/out-of-scope replies and clarification prompts without factual source claims.

