.PHONY: install run eval pull main
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

pull:
	ollama pull llama3
	ollama pull bge-m3

run:
	$(PYTHON) -m streamlit run app.py

eval:
	$(PYTHON) evaluate.py questions.json

main:
	$(PYTHON) main.py questions.json
