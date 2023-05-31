variable "DOCKER_REGISTRY" {
  default = "ghcr.io"
}
variable "DOCKER_ORG" {
  default = "darpa-askem"
}
variable "VERSION" {
  default = "local"
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

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

target "_platforms" {
  platforms = ["linux/amd64", "linux/arm64"]
}

target "funman-ibex" {
  context = "./ibex"
  dockerfile = "Dockerfile"
  tags = tag("funman-ibex", "", "")
}

target "funman-dreal4" {
  context = "."
  dockerfile = "Dockerfile.dreal4"
  tags = tag("funman-dreal4", "", "")
}

target "funman-base" {
  context = "./deploy/base"
  dockerfile = "Dockerfile"
  tags = tag("funman-base", "", "")
}

target "funman-pypi" {
  context = "./deploy/pypi"
  dockerfile = "Dockerfile"
  tags = tag("funman-pypi", "", "")
}

target "funman-git" {
  no-cache = true
  context = "./deploy/git"
  dockerfile = "Dockerfile"
  tags = tag("funman-git", "", "")
}

target "funman-api" {
  context = "./deploy/api"
  dockerfile = "Dockerfile"
  tags = tag("funman-api", "", "")
}
