.PHONY: test

# -----------------------------------------------------------------
#  Base Image Construction

OUTPUT_TYPE_LOCAL=$(OUTPUT_TYPE)
OUTPUT_TYPE_MULTI=$(OUTPUT_TYPE)

ifndef OUTPUT_TYPE
override OUTPUT_TYPE_LOCAL = load
override OUTPUT_TYPE_MULTI = push
endif

DOCKER_ORG_LOCAL=$(DOCKER_ORG)
DOCKER_ORG_MULTI=$(DOCKER_ORG)

ifndef DOCKER_ORG
override DOCKER_ORG_LOCAL = siftech
override DOCKER_ORG_MULTI = siftech
endif

REGISTRY?=localhost

DOCKER_BAKE=docker buildx bake -f ./docker/docker-bake.hcl

build-ibex: 
	DOCKER_ORG=${DOCKER_ORG_LOCAL} \
	DOCKER_REGISTRY=${REGISTRY} \
	${DOCKER_BAKE} funman-ibex --${OUTPUT_TYPE_LOCAL}

multiplatform-build-ibex:
	DOCKER_ORG=${DOCKER_ORG_MULTI} \
	${DOCKER_BAKE} funman-ibex-multiplatform --${OUTPUT_TYPE_MULTI}

build-dreal4: build-ibex
	DOCKER_ORG=${DOCKER_ORG_LOCAL} \
	DOCKER_REGISTRY=${REGISTRY} \
	${DOCKER_BAKE} funman-dreal4 --${OUTPUT_TYPE_LOCAL}

multiplatform-build-dreal4: multiplatform-build-ibex
	DOCKER_ORG=${DOCKER_ORG_MULTI} \
	${DOCKER_BAKE} funman-dreal4-multiplatform --${OUTPUT_TYPE_MULTI}

build-base: build-dreal4
	DOCKER_ORG=${DOCKER_ORG_LOCAL} \
	DOCKER_REGISTRY=${REGISTRY} \
	${DOCKER_BAKE} funman-base --${OUTPUT_TYPE_LOCAL}

multiplatform-build-base: multiplatform-build-dreal4
	DOCKER_ORG=${DOCKER_ORG_MULTI} \
	${DOCKER_BAKE} funman-base-multiplatform --${OUTPUT_TYPE_MULTI}

build-git: build-base
	DOCKER_ORG=${DOCKER_ORG_LOCAL} \
	DOCKER_REGISTRY=${REGISTRY} \
	${DOCKER_BAKE} funman-git --${OUTPUT_TYPE_LOCAL}

multiplatform-build-git: multiplatform-build-base
	DOCKER_ORG=${DOCKER_ORG_MULTI} \
	${DOCKER_BAKE} funman-git-multiplatform --${OUTPUT_TYPE_MULTI}

build-api: build-git
	DOCKER_ORG=${DOCKER_ORG_LOCAL} \
	DOCKER_REGISTRY=${REGISTRY} \
	${DOCKER_BAKE} funman-api --${OUTPUT_TYPE_LOCAL}

multiplatform-build-api: multiplatform-build-git
	DOCKER_ORG=${DOCKER_ORG_MULTI} \
	${DOCKER_BAKE} funman-api-multiplatform --${OUTPUT_TYPE_MULTI}

# -----------------------------------------------------------------
#  Development Container Utils

DREAL_LOCAL_REPO?=../dreal4

DEV_NAME:=funman-dev
DEV_TAG:=local-latest
DEV_TAGGED_NAME:=$(DEV_NAME):$(DEV_TAG)

DEV_REGISTRY:=localhost:5000
DEV_ORG:=siftech
DEV_IMAGE:=${DEV_REGISTRY}/${DEV_ORG}/${DEV_TAGGED_NAME}

DEV_CONTAINER:=funman-dev
MAKE:=make --no-print-directory

FUNMAN_BASE_PORT?=21000
ifeq (,$(wildcard .docker-as-root))
FUNMAN_SKIP_USER_ARGS=0
FUNMAN_DEV_TARGET=funman-dev
else
FUNMAN_SKIP_USER_ARGS=1
FUNMAN_DEV_TARGET=funman-dev-as-root
endif

FUNMAN_DEV_TARGET_ARCH=
ifdef TARGET_ARCH
override FUNMAN_DEV_TARGET_ARCH=--set=*.platform=$(TARGET_ARCH)
endif

FUNMAN_DEV_NO_CACHE=
ifdef NO_CACHE
override FUNMAN_DEV_NO_CACHE=--no-cache
endif


DEBUG_IBEX?=no

build-development-environment:
	DOCKER_ORG=${DEV_ORG} \
	DOCKER_REGISTRY=${REGISTRY} \
	$(if $(filter 0,$(FUNMAN_SKIP_USER_ARGS)),FUNMAN_DEV_UNAME=$(shell echo $$USER)) \
	$(if $(filter 0,$(FUNMAN_SKIP_USER_ARGS)),FUNMAN_DEV_UID=$(shell echo $$(id -u))) \
	$(if $(filter 0,$(FUNMAN_SKIP_USER_ARGS)),FUNMAN_DEV_GID=$(shell echo $$(id -g))) \
	${DOCKER_BAKE} ${FUNMAN_DEV_TARGET} --${OUTPUT_TYPE_LOCAL} ${FUNMAN_DEV_TARGET_ARCH} ${FUNMAN_DEV_NO_CACHE}

local-registry:
	docker start local_registry \
		|| docker run -d \
			--name local_registry \
			--network host \
			registry:2

use-docker-driver: local-registry
	docker buildx use funman-builder \
		|| docker buildx create \
			--name funman-builder \
			--use \
			--driver-opt network=host

dev: use-docker-driver
	OUTPUT_TYPE=push \
	REGISTRY=${DEV_REGISTRY} \
	$(MAKE) build-development-environment

dev-amd64:
	TARGET_ARCH=linux/amd64 $(MAKE) dev

dev-arm64:
	TARGET_ARCH=linux/arm64 $(MAKE) dev

debug: 
	DEBUG_IBEX=yes \
	OUTPUT_TYPE=push \
	REGISTRY=${DEV_REGISTRY} \
	$(MAKE) build-development-environment

run-dev-container:
	@FUNMAN_HOME=$(if $(filter 0,$(FUNMAN_SKIP_USER_ARGS)),/home/$$USER,/root) \
	&& if [ -e "$(DREAL_LOCAL_REPO)" ] ; then \
		DREAL_LOCAL_VOLUME_CMD=-v ; \
		DREAL_LOCAL_VOLUME_ARG=$$(realpath $(DREAL_LOCAL_REPO)):$$FUNMAN_HOME/dreal4:rw ; \
	else \
		echo "WARNING: Dreal4 repo not found at $(DREAL_LOCAL_REPO)" ; \
		DREAL_LOCAL_VOLUME_CMD= ; \
		DREAL_LOCAL_VOLUME_ARG= ; \
	fi \
	&& docker run \
		-d \
		-it \
		--cpus=5 \
		--cap-add=SYS_PTRACE \
		--name ${DEV_CONTAINER} \
		-p 127.0.0.1:$(FUNMAN_BASE_PORT):8888 \
		-p 127.0.0.1:$$(($(FUNMAN_BASE_PORT) + 1)):8190 \
		-v $$PWD:$$FUNMAN_HOME/funman:rw $$DREAL_LOCAL_VOLUME_CMD $$DREAL_LOCAL_VOLUME_ARG \
		${DEV_IMAGE}

launch-dev-container:
	@$(MAKE) delete-dev-container-if-out-of-date
	@docker container inspect ${DEV_CONTAINER} > /dev/null 2>&1 \
		|| $(MAKE) run-dev-container
	@test $$(docker container inspect -f '{{.State.Running}}' ${DEV_CONTAINER}) == 'true' > /dev/null 2>&1 \
		|| docker start ${DEV_CONTAINER}
	@docker attach ${DEV_CONTAINER}

delete-dev-container:
	@echo "Deleting dev container:"
	docker stop ${DEV_CONTAINER}
	docker rm ${DEV_CONTAINER}

pull-dev-container:
ifndef TARGET_ARCH
	docker pull ${DEV_IMAGE}
else
	echo "Using arch:" ${TARGET_ARCH} 
	docker pull --platform ${TARGET_ARCH} ${DEV_IMAGE}
endif

delete-dev-container-if-out-of-date: pull-dev-container
	@if (docker container inspect ${DEV_CONTAINER} > /dev/null 2>&1) ; then \
		FUNMAN_CONTAINER_SHA=$$(docker inspect -f '{{.Image}}' ${DEV_CONTAINER}) ; \
		FUNMAN_IMAGE_SHA=$$(docker images --no-trunc --quiet ${DEV_IMAGE}) ; \
		if [ "$$FUNMAN_IMAGE_SHA" != "$$FUNMAN_CONTAINER_SHA" ] ; then \
		  echo "Dev container out of date:" ; \
			echo "  Container: $$FUNMAN_CONTAINER_SHA" ; \
			echo "  Image: $$FUNMAN_IMAGE_SHA" ; \
		  $(MAKE) delete-dev-container ; \
		fi \
	fi

docker-as-root:
	touch .docker-as-root

# -----------------------------------------------------------------
#  API Client
generate-api-client:
	pip install openapi-python-client
	openapi-python-client --install-completion
	uvicorn funman.api.api:app --host 0.0.0.0 --port 8190 &
	sleep 1
	openapi-python-client generate --url http://0.0.0.0:8190/openapi.json
	pip install -e funman-api-client

update-api-client:
	openapi-python-client update --url http://0.0.0.0:8190/openapi.json

# -----------------------------------------------------------------
#  Utils
venv:
	test -d .venv || python -m venv .venv
	source .venv/bin/activate && pip install -Ur requirements-dev.txt
	source .venv/bin/activate && pip install -Ur requirements-dev-extras.txt

format:
	pycln --config pyproject.toml .
	isort --settings-path pyproject.toml .
	black --config pyproject.toml .
	
FUNMAN_VERSION ?= 0.0.0
CMD_UPDATE_VERSION_PY = sed -i -E 's/^__version__ = \"[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]*)?\"/__version__ = \"${FUNMAN_VERSION}\"/g'
CMD_UPDATE_VERSION_SH = sed -i -E 's/^FUNMAN_VERSION=\"[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]*)?\"/FUNMAN_VERSION=\"${FUNMAN_VERSION}\"/g'
update-versions:
	@test "${FUNMAN_VERSION}" != "0.0.0" || (echo "ERROR: FUNMAN_VERSION must be set" && exit 1)
	@${CMD_UPDATE_VERSION_PY} auxiliary_packages/funman_demo/src/funman_demo/_version.py
	@${CMD_UPDATE_VERSION_PY} auxiliary_packages/funman_dreal/src/funman_dreal/_version.py
	@${CMD_UPDATE_VERSION_PY} src/funman/_version.py
	@${CMD_UPDATE_VERSION_SH} terarium/scripts/run-api-in-docker.sh

# -----------------------------------------------------------------
#  Distribution
dist: update-versions
	mkdir -p dist
	mkdir -p dist.bkp
	rsync -av --ignore-existing --remove-source-files dist/ dist.bkp/
	python -m build --outdir ./dist .
	python -m build --outdir ./dist auxiliary_packages/funman_demo
	python -m build --outdir ./dist auxiliary_packages/funman_dreal

check-test-release: dist
	@echo -e "\nReleasing the following packages to TestPyPI:"
	@ls -1 dist | sed -e 's/^/    /'
	@echo -n -e "\nAre you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

test-release: check-test-release
	python3 -m twine upload --repository testpypi dist/*

test:
	pytest test

check-release: dist
	@echo -e "\nReleasing the following packages to PyPI:"
	@ls -1 dist | sed -e 's/^/    /'
	@echo -n -e "\nAre you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

release: check-release
	python3 -m twine upload dist/

# -----------------------------------------------------------------
#  Docs and Pages
#  TODO: This can probably be automated
DOCS_REMOTE ?= origin

docs:
	sphinx-apidoc -f -o ./docs/source ./src/funman -t ./docs/apidoc_templates --no-toc  
	mkdir -p ./docs/source/_static
	mkdir -p ./docs/source/_templates
	pyreverse \
		-k \
		-d ./docs/source/_static \
		./src/funman
	cd docs && $(MAKE) clean html

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

generate-amr-classes:
	datamodel-codegen --input-file-type jsonschema  --url https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/base_schema.json --output src/funman/model/generated_models/base_schema.py
	#datamodel-codegen --input-file-type jsonschema --url https://raw.githubusercontent.com/danbryce/Model-Representations/main/metadata_schema.json --output src/funman/model/generated_models/metadata_schema.py
	datamodel-codegen --input-file-type jsonschema --url https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/petrinet_schema.json  --output src/funman/model/generated_models/petrinet_schema.py
	datamodel-codegen --input-file-type jsonschema --url https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/regnet_schema.json  --output src/funman/model/generated_models/regnet_schema.py
