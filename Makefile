format:
	python -m black flexs
	python -m isort --profile black flexs

lint-strict:
	python -m pylint --reports=n flexs
	python -m pydocstyle flexs

lint:
	python -m flake8 flexs --exit-zero

test:
	python -m pytest tests
