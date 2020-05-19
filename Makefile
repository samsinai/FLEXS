# directories = ./explorers ./evaluators ./environments
directories = ./explorers

format:
	python -m black $(directories)
	python -m isort -rc $(directories)

lint:
	python -m pylint --reports=n --rcfile=pylintrc $(directories)
	python -m pydocstyle
