# Architecture (P00)

## Verified requirements

The knowledge base is only the supplied 12 government-service TXT files: passport, family book, and smart ID, each for first issue, renewal, lost, and damaged. Text may be cleaned or restructured only without changing facts. No other knowledge source is allowed.

The evaluator places questions.json beside main.py and runs:

    python main.py questions.json

The program must create a UTF-8 valid-JSON answers.json in that same directory and run offline. Answers must be generated at run time by the provided Ollama model. No pre-written answers, other AI models, or online AI services are allowed. Local embedding/reranking models are allowed, but all packages and models must be installed/downloaded in advance. The brief says the shared Ollama model is about 7B but gives no model tag.

## Boundaries

    main.py / CLI -> orchestrator -> corpus/original-excerpt store
                                  -> retrieval -> per-chat state
                                  -> Ollama generator -> output mapper/writer

The orchestrator preserves input structure, processes messages in order, contains errors per message, and writes the final file. Corpus/index code reads only the TXT files. It may normalize and index chunks, but each unit retains its verbatim original excerpt and source filename. Retrieval ranks indexed text but returns original evidence.

The generator receives a question, evidence, and only same-chat history. The output mapper emits evidence actually used as official context strings; it never substitutes scores, filenames, prompts, or generated summaries.

## History and evidence

Process chats and their messages in listed order. Initialize history for every chat_id and never transfer it between chats: the brief says each chat is a new user. Later messages receive only prior completed turns from their chat; append the current turn after recording its answer.

For a clear in-scope question, constrain the response to evidence and ensure emitted contexts support it. For unsupported information, say it is unavailable; for unclear service references, request clarification.

1. Ingest evidence-sized verbatim excerpts and filename metadata.
2. Retrieve and rank excerpts.
3. Generate from selected text plus scoped history.
4. Write only the used, exact text values as contexts, preserving relevant order and optionally deduplicating identical strings.

Official contexts is a list of strings, not retrieval objects. Source and score are internal diagnostics.

## Unresolved assumptions and risks

- The files specify no Ollama tag/endpoint, context size, timeout, hardware, dependency manager, or virtual environment.
- Chunking, retrieval/reranking, thresholds, prompts, answer style, and context limits are not prescribed. top_k=3 is only the team interface default.
- There is no machine-readable schema or stated behavior for malformed JSON, unknown keys, duplicate IDs, or empty lists.
- The example has concise/ellipsized context display while the brief permits cleaned/raw source pieces. This contract uses exact original excerpts; a different trimming policy needs team agreement.

## P06 evidence grounding

`generate_answer(question, contexts, history=None)` keeps its existing interface.
The local Ollama model selects exact, complete-line Arabic excerpts using JSON
with a supported/abstain status and evidence IDs. The generator validates every
selection against its referenced current retrieval text and returns only those
excerpts, with duplicate selections removed. It rejects malformed output,
unsupported text, missing evidence, and partial-line selections with the Arabic
insufficient-evidence reply. Retrieval inputs and original excerpt strings are
never modified. No extra model, dependency, or network service is introduced.

The system policy requires abstention for irrelevant, insufficient, conflicting,
or ambiguous evidence, and requires coverage of every requested service and
question part. Fees, documents, eligibility, deadlines, and procedures must be
explicitly stated for the requested service and case. History is serialized as
conversation-only data without prior context excerpts or assistant chat roles.
Retrieved text is also serialized as untrusted data; embedded instructions must
not be followed. Source filenames and ranking scores are not factual evidence.

This deliberately favors verbatim answers over fluent synthesis. Exact matching
prevents newly invented wording, but does not prove relevance, full coverage,
absence of contradictory passages, or safe interpretation of injected text.
Those semantic decisions still depend on the local model and retrieval quality;
zero hallucinations are not guaranteed. Mocked tests verify boundary handling,
output validation and policy construction, not real-model semantic reliability.
The caller still owns mapping actually used source excerpts to official contexts;
the string-only generation API does not expose selection IDs to that mapper.
