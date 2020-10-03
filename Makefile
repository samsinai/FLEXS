format:
	python -m black flexs
	python -m isort --profile black flexs

lint:
	python -m flake8 flexs --exit-zero

test:
	python -m pytest tests
