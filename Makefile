format:
	python -m black flexs
	python -m isort --profile black flexs

lint:
	python -m pylint --reports=n flexs
	python -m pydocstyle flexs
