.PHONY: install download-data run-extraction  run-similarity

all: install download-data run-extraction  run-similarity

install:
	poetry install

download-data:
	poetry run python ingestion/dataset-ingestion.py

run-extraction:
	poetry run python features/features_extraction.py

run-similarity:
	poetry run python similarity/check_similarity.py