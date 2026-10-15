# Prompt 2 - chunking update

Use a custom RecursiveCharacterTextSplitter for Arabic procedural docs:
chunk_size=1200, chunk_overlap=200, separators strictly ["\n\n", "\n", ".", " "]
so procedural steps (1, 2, 3) are never cut in half. Everything else unchanged.
