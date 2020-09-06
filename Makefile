format:
	python -m black flexs
	python -m isort -rc flexs

lint:
	python -m pylint --reports=n --rcfile=pylintrc flexs
	python -m pydocstyle flexs
