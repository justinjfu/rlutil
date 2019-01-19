export SHELL := /bin/bash

test:
	python -m unittest discover .

build:
	python setup.py build_ext --inplace 

