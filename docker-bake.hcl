variable "DOCKER_REGISTRY" {
  default = "ghcr.io"
}
variable "DOCKER_ORG" {
  default = "darpa-askem"
}
variable "VERSION" {
  default = "local"
}
variable "IBEX_BRANCH" {
  default = "ibex-2.8.5_using_mathlib-2.1.1"
}
variable "BAZEL_VERSION" {
  default = "6.0.0"
}
variable "DREAL_COMMIT_TAG" {
  default = "cf35b198bd3e9a9366740c6c9e244955a81bc229"
}
variable "AUTOMATES_COMMIT_TAG" {
  default = "e5fb635757aa57007615a75371f55dd4a24851e0"
}

# ----------------------------------------------------------------------------------------------------------------------

function "tag" {
  params = [image_name, prefix, suffix]
  result = [ "${DOCKER_REGISTRY}/${DOCKER_ORG}/${image_name}:${check_prefix(prefix)}${VERSION}${check_suffix(suffix)}" ]
}

function "check_prefix" {
  params = [tag]
  result = notequal("",tag) ? "${tag}-": ""
}

function "check_suffix" {
  params = [tag]
  result = notequal("",tag) ? "-${tag}": ""
}

function "compose_registry" {
  params = [registry, org]
  result = notequal("localhost", registry) ? "${registry}/${org}/" : "${registry}/${org}/"
}

# ----------------------------------------------------------------------------------------------------------------------

group "api-git" {
    targets = ["funman-ibex", "funman-dreal4", "funman-base", "funman-git", "funman-api"]
}

# ----------------------------------------------------------------------------------------------------------------------

target "_platforms" {
  platforms = ["linux/amd64", "linux/arm64"]
}

target "funman-ibex" {
  context = "./ibex"
  args = {
    IBEX_BRANCH = "${IBEX_BRANCH}"
  }
  dockerfile = "Dockerfile"
  tags = tag("funman-ibex", "", "${IBEX_BRANCH}")
  output = ["type=cacheonly"]
}

target "funman-dreal4" {
  context = "."
  contexts = {
    baseapp = "target:funman-ibex"
  }
  args = {
    SIFT_REGISTRY_ROOT = compose_registry("${DOCKER_REGISTRY}","${DOCKER_ORG}")
    IBEX_TAG = "${VERSION}-${IBEX_BRANCH}"
    BAZEL_VERSION = "${BAZEL_VERSION}" 
    DREAL_COMMIT_TAG = "${DREAL_COMMIT_TAG}"
  }
  dockerfile = "Dockerfile.dreal4"
  tags = tag("funman-dreal4", "", "${DREAL_COMMIT_TAG}")
  output = ["type=cacheonly"]
}

target "funman-base" {
  context = "./deploy/base"
  contexts = {
    baseapp = "target:funman-dreal4"
  }
  args = {
    SIFT_REGISTRY_ROOT = "${DOCKER_REGISTRY}/${DOCKER_ORG}/"
    DREAL_TAG = "${VERSION}-${DREAL_COMMIT_TAG}"
    AUTOMATES_COMMIT_TAG = "${AUTOMATES_COMMIT_TAG}"
  }
  dockerfile = "Dockerfile"
  tags = tag("funman-base", "", "${AUTOMATES_COMMIT_TAG}")
  output = ["type=cacheonly"]
}

target "funman-pypi" {
  context = "./deploy/pypi"
  contexts = {
    baseapp = "target:funman-base"
  }
  args = {
    SIFT_REGISTRY_ROOT = "${DOCKER_REGISTRY}/${DOCKER_ORG}/"
    FROM_TAG = "${VERSION}-${AUTOMATES_COMMIT_TAG}"
  }
  dockerfile = "Dockerfile"
  tags = tag("funman-pypi", "", "")
}

target "funman-git" {
  context = "."
  contexts = {
    baseapp = "target:funman-base"
  }
  args = {
    SIFT_REGISTRY_ROOT = "${DOCKER_REGISTRY}/${DOCKER_ORG}/"
    FROM_TAG = "${VERSION}-${AUTOMATES_COMMIT_TAG}"
  }
  dockerfile = "./deploy/git/Dockerfile"
  tags = tag("funman-git", "", "")
  output = ["type=cacheonly"]
}

target "funman-api" {
  context = "./deploy/api"
  contexts = {
    baseapp = "target:funman-git"
  }
  args = {
    SIFT_REGISTRY_ROOT = "${DOCKER_REGISTRY}/${DOCKER_ORG}/"
    FROM_IMAGE = "funman-git"
    FROM_TAG = "${VERSION}"
  }
  dockerfile = "Dockerfile"
  tags = tag("funman-api", "", "")
}
