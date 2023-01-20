ARG SIFT_REGISTRY_ROOT
ARG FROM_TAG=${TARGETOS}-${TARGETARCH}
FROM ${SIFT_REGISTRY_ROOT}funman-base:${FROM_TAG}

# Install funman dev packages
RUN pip install --no-cache-dir \
  funman \
  funman_demo \
  funman_dreal

WORKDIR "/"
