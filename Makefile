PIPENV=PIPENV_VENV_IN_PROJECT=1 pipenv

.PHONY: setup-dev-env destroy-dev-env
setup-dev-env: setup-pipenv setup-pysmt

setup-pipenv:
	# Initializing virtual environment
	$(PIPENV) install --dev

setup-pysmt:
	# Installing Z3
	$(PIPENV) run pysmt-install --z3 \
		--confirm-agreement \
		--install-path ./.smt-solvers
	$(PIPENV) run pysmt-install --check

destroy-dev-env:
	$(PIPENV) --rm

setup-conda-dev-env: set-conda setup-conda-packages setup-pipenv setup-pysmt 

set-conda:
	$(PIPENV) --python=$(shell conda run which python) --site-packages

setup-conda-packages:
	$(PIPENV) run conda install scipy pygraphviz scikit-learn igraph  python-igraph lxml Pillow coverage psutil