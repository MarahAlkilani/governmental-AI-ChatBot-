# Prompt 1 - original task

Act as a Principal AI Engineer and a 42 School Hackathon Grand Prize Winner. The 4-hour hackathon has officially started. We received the `participant_pack` (no starter code):
1. `data/` - 12 Arabic .txt files about Passport, ID and Family Book procedures.
2. `questions.json` - test cases.
3. `answers_example.json` - expected output format.

Goal: 100% stability, zero hallucinations, perfect evaluation score, keep the "Kashaf" (Searchlight) UI.

Deliver 4 files:
1. `rag_engine.py` - llama3 (LLM) + bge-m3 (embeddings) via local Ollama; load the 12 Arabic files from data/; Arabic-optimised chunking so numbered steps are never cut; anti-hallucination prompt: answer ONLY from context, otherwise output exactly: "عذراً، المعلومات غير متوفرة في الوثائق."
2. `evaluate.py` - loads questions.json, runs each question through rag_engine, writes answers.json (same structure as answers_example.json), hacker-style rich progress bar, ends with "Evaluation Complete ✅".
3. `app.py` - Streamlit UI, theme "Kashaf Searchlight" (Midnight Blue #0F172A, Amber #F59E0B, no harsh neon). BUG FIX: do NOT hide the sidebar toggle (`[data-testid="collapsedControl"]`). KILLER FEATURE: numbered steps (1. 2. 3. / أولاً، ثانياً) rendered as a "Visual Procedure Timeline" via st.markdown(unsafe_allow_html=True); no st.components.v1.html.
4. `Makefile` - make install / make run / make eval.
