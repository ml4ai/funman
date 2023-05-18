# Building Docker Images

The funman docker image is built up from several layers of images.

The general order (and purpose) is:
1. ubuntu:20.04  (use the default dockerhub ubuntu image.)
2. funman-ibex   (build IBEX from source with minor tweaks.)
3. funman-dreal  (build a modified version of Dreal from source.)
4. funman-base   (base image with all funman dependencies.)
5. Either a or b:
  a. funman-pypi (image with funman installed via pypi.)
  b. funman-git  (image with funman installed via a git branch. Defaults to main.)
6. funman-api    (image to host funman api using uvicorn.)

Each image is briefly described below. General build commands are also listed.
Note that there are more specific build commands in the Makefile but they use
a multitude of args for handling platform specification and various other details
related to local development containers.

## General notes on building images:
Docker buildx has some quirks when attempting to use it 
to build a series of images in a local `docker images` cache. We use a 
local registry to tag and push images from docker buildx so that later
images built with docker buildx can easily extend those images.

For example:
```
docker tag funman-ibex:linux-arm64 localhost:5000/sift/funma-ibex:linux-arm64
docker push localhost:5000/sift/funma-ibex:linux-arm64
```

Depending on use case and build tools there may be better options described here:
https://docs.docker.com/engine/reference/commandline/buildx_build/

---

## funman-ibex
Dockerfile path: ibex/Dockerfile
### Description
Install dependencies for IBEX then build and install from source.
## Build Command
From the root of the repo:
```
DOCKER_BUILDKIT=1 docker buildx build \
	--output "type=docker" \
	--platform linux/arm64 \
	--tag funman-ibex:linux-arm64 \
	-f ./ibex/Dockerfile ./ibex
```

## funman-dreal
Dockerfile path: Dockerfile.dreal4
### Description
Install everything needed to build Dreal from source.
Also build and install from source.
### Args
- IBEX_TAG:
  Tag to use when extending the funman-ibex image
## Build Command
From the root of the repo:
```
DOCKER_BUILDKIT=1 docker buildx build \
	--output "type=docker" \
	--platform linux/arm64 \
	--build-arg IBEX_TAG=linux-arm64 \
	--tag funman-dreal:linux-arm64 \
	-f ./Dockerfile.dreal4 .
```

## funman-base
Dockerfile path: deploy/base/Dockerfile
### Description
Install the funman dependencies (but not funman) into the image.
Intended to be the base to allow creation of variant funman images.
### Args
- DREAL_TAG:
  Tag to use when extending the funman-dreal image
## Build Command
From the root of the repo:
```
DOCKER_BUILDKIT=1 docker buildx build \
	--output "type=docker" \
	--platform linux/arm64 \
	--build-arg DREAL_TAG=linux-arm64 \
	--tag funman-base:linux-arm64 \
	./deploy/base
```

## funman-pypi
Dockerfile path: deploy/pypi/Dockerfile
### Description
Install the funman packges into the image using the latest version visible on pypi
### Args
- FROM_TAG:
  Tag to use on when extending the funman-base image
## Build Command
From the root of the repo:
```
DOCKER_BUILDKIT=1 docker buildx build \
	--output "type=docker" \
	--platform linux/arm64 \
	--build-arg FROM_TAG=linux-arm64 \
	--tag funman-pypi:linux-arm64 \
	./deploy/pypi
```

## funman-git
Dockerfile path: deploy/git/Dockerfile
### Description
Shallow clone the repo from https://github.com/ml4ai/funman.git at the
branch specified by the FUNMAN_BRANCH arg. Then install the funman packages
into the image environment.
### Args
- FUNMAN_BRANCH:
  Branch to use when shallow cloning the funman repo
- FROM_TAG:
  Tag to use on when extending the funman-base image
## Build Command
From the root of the repo:
```
DOCKER_BUILDKIT=1 docker buildx build \
	--output "type=docker" \
	--platform linux/arm64 \
	--build-arg FROM_TAG=linux-arm64 \
	--build-arg FUNMAN_BRANCH=main \
	--tag funman-git:linux-arm64 \
	./deploy/git
```

## funman-api
Dockerfile path: deploy/api/Dockerfile
### Description 
Installs unvicorn and updates the CMD to run the api server.
### Args
- FROM_IMAGE:
  Image to build the api image on top of.
- FROM_TAG:
  Tag to use on FROM_IMAGE
## Build Command
From the root of the repo:
```
DOCKER_BUILDKIT=1 docker buildx build \
	--output "type=docker" \
	--platform linux/arm64 \
	--build-arg FROM_IMAGE=funman-git \
	--build-arg FROM_TAG=linux-arm64 \
	--tag funman-api:linux-arm64 \
	./deploy/git
```
  

