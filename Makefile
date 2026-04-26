.PHONY: install clean download-data run-extraction run-similarity test verify all

all: install download-data run-extraction run-similarity

install:
	poetry install

clean:
	rm -rf results/*


download-data:
	poetry run python ingestion/dataset-ingestion.py

run-extraction:
	poetry run python features/features_extraction.py

run-similarity:
	poetry run python similarity/check_similarity.py

test:
	poetry run pytest tests/ -v --tb=short

verify:
	poetry run python verify_prototype.py
