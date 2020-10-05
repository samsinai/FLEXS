format:
	python -m black .
	python -m isort --profile black .

lint:
	python -m flake8 flexs

test:
	python -m pytest tests

.PHONY: docs
docs:
	cd ./docs && make html