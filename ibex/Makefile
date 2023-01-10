
TARGET_OS?=linux
TARGET_ARCH?=amd64

IMAGE_NAME=funman-ibex
IMAGE_TAG=$(TARGET_OS)-$(TARGET_ARCH)
BUILD_FLAGS?=

build-ibex-image:

build-ibex-images:
	for arch in arm64 amd64 ; do \
		make build-ibex-image TARGET_OS=linux TARGET_ARCH=$$arch ; \
	done

