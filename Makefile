PIPENV=PIPENV_VENV_IN_PROJECT=1 pipenv
DOCS_REMOTE?=origin

.PHONY: setup-dev-env destroy-dev-env docs
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
	$(PIPENV) run conda install scipy pygraphviz scikit-learn lxml Pillow coverage psutil

docs:
	sphinx-apidoc -f -o ./docs/source ./src/funman -t ./docs/apidoc_templates --no-toc --module-first
	mkdir -p ./docs/source/_static
	mkdir -p ./docs/source/_templates
	pyreverse ./src/funman -d ./docs/source/_static
	cd docs && make clean html

init-pages:
	@if [ -n "$$(git ls-remote --exit-code $(DOCS_REMOTE) gh-pages)" ]; then echo "GitHub Pages already initialized"; exit 1; fi;
	git switch --orphan gh-pages
	git commit --allow-empty -m "initial pages"
	git push -u $(DOCS_REMOTE) gh-pages
	git checkout main
	git branch -D gh-pages

deploy-pages: docs
	mv docs/build/html www
	touch www/.nojekyll
	rm www/.buildinfo || true
	git checkout gh-pages
	rm -r docs || true
	mv www docs
	git add -f docs
	git commit -m "update pages" || true
	git push $(DOCS_REMOTE)
	git checkout main
	git branch -D gh-pages