
TARGET_OS?=linux
TARGET_ARCH?=amd64

IMAGE_NAME=sift/funman-ibex

build-ibex-image:
	DOCKER_BUILDKIT=1 docker buildx build \
		--output "type=image" \
		--platform $(TARGET_OS)/$(TARGET_ARCH) \
		--tag $(IMAGE_NAME):$(TARGET_OS)-$(TARGET_ARCH) \
		-f ./Dockerfile .

build-ibex-images:
	for arch in arm64 amd64 ; do \
		make build-ibex-image TARGET_OS=linux TARGET_ARCH=$$arch ; \
	done

