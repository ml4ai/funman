# -----------------------------------------------------------------
#  Deployment

OUTPUT_TYPE_LOCAL=$(OUTPUT_TYPE)
OUTPUT_TYPE_MULTI=$(OUTPUT_TYPE)

ifndef OUTPUT_TYPE
override OUTPUT_TYPE_LOCAL = load
override OUTPUT_TYPE_MULTI = push
endif

build-ibex: 
	DOCKER_REGISTRY=${REGISTRY} docker buildx bake funman-ibex --${OUTPUT_TYPE_LOCAL}

multiplatform-build-ibex:
	docker buildx bake funman-ibex-multiplatform --${OUTPUT_TYPE_MULTI}

build-dreal4: build-ibex
	DOCKER_REGISTRY=${REGISTRY} docker buildx bake funman-dreal4 --${OUTPUT_TYPE_LOCAL}

multiplatform-build-dreal4: multiplatform-build-ibex
	docker buildx bake funman-dreal4-multiplatform --${OUTPUT_TYPE_MULTI}

build-base: build-dreal4
	DOCKER_REGISTRY=${REGISTRY} docker buildx bake funman-base --${OUTPUT_TYPE_LOCAL}

multiplatform-build-base: multiplatform-build-dreal4
	docker buildx bake funman-base-multiplatform --${OUTPUT_TYPE_MULTI}

build-git: build-base
	DOCKER_REGISTRY=${REGISTRY} docker buildx bake funman-git --${OUTPUT_TYPE_LOCAL}

multiplatform-build-git: multiplatform-build-base
	docker buildx bake funman-git-multiplatform --${OUTPUT_TYPE_MULTI}

build-api: build-git
	DOCKER_REGISTRY=${REGISTRY} docker buildx bake funman-api --${OUTPUT_TYPE_LOCAL}

multiplatform-build-api: multiplatform-build-git
	docker buildx bake funman-api-multiplatform --${OUTPUT_TYPE_MULTI}

# -----------------------------------------------------------------
#  Development environment 

REGISTRY?=localhost
DREAL_LOCAL_REPO?=../dreal4

DEV_NAME:=funman-dev
DEV_TAG:=local-latest
DEV_TAGGED_NAME:=$(DEV_NAME):$(DEV_TAG)

DEV_REGISTRY:=localhost:5000
DEV_ORG:=darpa-askem
DEV_IMAGE:=${DEV_REGISTRY}/${DEV_ORG}/${DEV_TAGGED_NAME}

DEV_CONTAINER:=funman-dev
MAKE:=make --no-print-directory

FUNMAN_BASE_PORT?=21000
ifeq (,$(wildcard .docker-as-root))
FUNMAN_SKIP_USER_ARGS=0
FUNMAN_USER=$(shell echo $$USER)
FUNMAN_DEV_DOCKERFILE=Dockerfile
else
FUNMAN_SKIP_USER_ARGS=1
FUNMAN_USER=
FUNMAN_DEV_DOCKERFILE=Dockerfile.root
endif

build-development-environment: build-dreal4
	DOCKER_REGISTRY=${REGISTRY} docker buildx bake funman-dev --${OUTPUT_TYPE_LOCAL}

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
	$(MAKE) build-development-environment \
	&& echo docker pull $${REGISTRY}/darpa-askem/${DEV_TAGGED_NAME}

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
	docker pull ${DEV_IMAGE}

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
