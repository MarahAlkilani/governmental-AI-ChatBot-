# Arabic Government Services RAG

Offline Arabic answers using organizer-provided documents and the local Ollama model `command-r7b-arabic:7b`.

## Setup on WSL Ubuntu

Run these commands from the project root. Complete installation while internet access is available; reuse the existing `.venv` if already set up.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If Ubuntu reports that venv is unavailable, install its `python3-venv` package first.

Install Ollama using the [official Linux instructions](https://docs.ollama.com/linux). Keep its local server running at `http://localhost:11434`. If the service is not already running, run `ollama serve` in another terminal. Verify connectivity and the exact model tag:

```bash
ollama list
```

The list must contain `command-r7b-arabic:7b`. The submission machine already has it installed. On a new machine, provision that exact organizer-approved model before going offline; do not substitute another model.

## Organizer files

Place the 12 original service TXT files in `data/` beside `main.py`. Alternatively, keep the supplied package layout:

```text
hackathon_participant_pack/participant_pack/data/*.txt
hackathon_participant_pack/participant_pack/questions.json
```

The loader searches `hackathon_participant_pack/` recursively first and uses root `data/` only when that package contains no TXT files. Use one intended corpus location to avoid unintended duplicate or extra documents. Preserve filenames and original text. Organizer files are excluded from Git and must be supplied separately on a fresh checkout.

## Run

With the environment activated, export the model name in the same terminal before running:

```bash
export OLLAMA_MODEL=command-r7b-arabic:7b
python main.py questions.json
```

For the supplied practice input:

```bash
python main.py hackathon_participant_pack/participant_pack/questions.json
```

The CLI writes UTF-8 `answers.json` **beside the input file**, replacing any prior output there. Root `questions.json` produces root `answers.json`; the nested practice command produces `hackathon_participant_pack/participant_pack/answers.json`. A root copy of the practice output is not refreshed by the nested command.

Input is an object containing `chats`, each with `chat_id` and `messages`; each message contains `message_id` and a string `question`. Output preserves these fields and adds an Arabic string `answer` and a list of exact source-text strings `contexts` to every message. Abstentions and local model failures have empty contexts. See `docs/contracts.md` for the full interface contract.

After dependencies and the model are installed, execution is fully offline: requests go only to local Ollama, with no cloud APIs or runtime downloads. History stays within each chat. The model selects source lines, which Python copies verbatim. Relevance and completeness still depend on retrieval and model judgment; service-free follow-ups can abstain. Local model errors appear on stderr. `OLLAMA_TIMEOUT_SECONDS` optionally overrides the default 45-second timeout.

## Tests

In the activated environment, install `pytest` beforehand if needed (it is not listed in runtime requirements), then run:

```bash
python -m pytest -q
```
