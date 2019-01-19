export SHELL := /bin/bash

test:
	python -m unittest discover .

unittests:
	pytest rlutil

coverage:
	pytest --cov=rlutil --cov-config=.coveragerc rlutil

build:
	python setup.py build_ext --inplace 

