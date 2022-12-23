DOCS_REMOTE ?= origin
DEV_CONTAINER ?= funman-dev
DEV_TAG ?= funman-dev
DEPLOY_TAG ?= funman

USING_PODMAN := $(shell docker --version 2> /dev/null | grep -q podman && echo 1 || echo 0)

.PHONY: docs

venv:
	test -d .venv || python -m venv .venv
	source .venv/bin/activate && pip install -Ur requirements-dev.txt
	source .venv/bin/activate && pip install -Ur requirements-dev-extras.txt

docs:
	sphinx-apidoc -f -o ./docs/source ./src/funman -t ./docs/apidoc_templates --no-toc --module-first
	mkdir -p ./docs/source/_static
	mkdir -p ./docs/source/_templates
	pyreverse \
		-k \
		-d ./docs/source/_static \
		./src/funman
	cd docs && make clean html

init-pages:
	@if [ -n "$$(git ls-remote --exit-code $(DOCS_REMOTE) gh-pages)" ]; then echo "GitHub Pages already initialized"; exit 1; fi;
	git switch --orphan gh-pages
	git commit --allow-empty -m "initial pages"
	git push -u $(DOCS_REMOTE) gh-pages
	git checkout main
	git branch -D gh-pages

deploy-pages:
	mv docs/build/html www
	touch www/.nojekyll
	rm www/.buildinfo || true
	git checkout --track $(DOCS_REMOTE)/gh-pages
	rm -r docs || true
	mv www docs
	git add -f docs
	git commit -m "update pages" || true
	git push $(DOCS_REMOTE)
	git checkout main
	git branch -D gh-pages

build-docker-dreal:
	docker build -t funman_dreal4 -f ./Dockerfile.dreal4 .

build-docker: build-docker-dreal
	DOCKER_BUILDKIT=1 docker build \
		--build-arg UNAME=$$USER \
		--build-arg UID=$$(id -u) \
		--build-arg GID=$$(id -g) \
		-t ${DEV_TAG} -f ./Dockerfile .

build:
	make build-docker

build-for-deployment: build-docker-dreal
	DOCKER_BUILDKIT=1 docker build \
		-t ${DEPLOY_TAG} -f ./Dockerfile.deploy .

run-deployment-image:
	docker run -it --rm -p 127.0.0.1:8888:8888 ${DEPLOY_TAG}:latest

run:
	@test "${USING_PODMAN}" == "1" && make run-podman || make run-docker

run-docker:
	docker run \
		-d \
		-it \
		--cpus=8 \
		--name ${DEV_CONTAINER} \
    -p 127.0.0.1:8888:8888 \
		-v $$PWD:/home/$$USER/funman \
		${DEV_TAG}:latest

run-podman:
	podman run \
		-d \
		-it \
		--cpus=8 \
		--name ${DEV_CONTAINER} \
		--user $$USER \
		-p 127.0.0.1:8888:8888 \
		-v $$PWD:/home/$$USER/funman \
		--userns=keep-id \
		${DEV_TAG}:latest

launch-dev-container:
	@docker container inspect ${DEV_CONTAINER} > /dev/null 2>&1 \
		|| make run
	@test $(shell docker container inspect -f '{{.State.Running}}' ${DEV_CONTAINER}) == 'true' > /dev/null 2>&1 \
		|| docker start ${DEV_CONTAINER}
	@docker attach ${DEV_CONTAINER}

rm-dev-container:
	@docker container rm ${DEV_CONTAINER}

install-pre-commit-hooks:
	@pre-commit install

format:
	pycln --config pyproject.toml .
	isort --settings-path pyproject.toml .
	black --config pyproject.toml .
