.PHONY: install clean download-data download-artifact run-extraction run-similarity run-wasserstein run-fmd run-euclidean run-correlation test verify all

all: install download-data run-extraction run-similarity

install:
	poetry install

clean:
	rm -rf results/*


download-data:
	poetry run python ingestion/dataset-ingestion.py

download-artifact:
	poetry run python ingestion/download_artifact.py

run-extraction:
	poetry run python features/features_extraction.py

run-similarity:
	poetry run python similarity/check_similarity.py

test:
	poetry run pytest tests/ -v --tb=short

verify:
	poetry run python verify_prototype.py

run-wasserstein:
	poetry run python similarity/wasserstein.py

run-euclidean:
	poetry run python similarity/euclidean.py

SAMPLE ?=

run-fmd:
	poetry run python fmd/compute_fmd.py $(if $(SAMPLE),--sample $(SAMPLE),)

run-correlation:
    poetry run python analysis/correlation.py

