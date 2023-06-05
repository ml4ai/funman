ARG SIFT_REGISTRY_ROOT
ARG FROM_TAG=${TARGETOS}-${TARGETARCH}
FROM ${SIFT_REGISTRY_ROOT}funman-base:${FROM_TAG}

ARG FUNMAN_BRANCH=main

#RUN git clone --depth=1 --branch ${FUNMAN_BRANCH} https://github.com/ml4ai/funman.git /funman
COPY ./setup.py /funman/setup.py
COPY ./pyproject.toml /funman/pyproject.toml
COPY ./README.md /funman/README.md
COPY ./src /funman/src
COPY ./auxiliary_packages/funman_demo /funman/auxiliary_packages/funman_demo
COPY ./auxiliary_packages/funman_dreal /funman/auxiliary_packages/funman_dreal

# Install funman dev packages
RUN pip install -I \
  /funman \
  /funman/auxiliary_packages/funman_demo \
  /funman/auxiliary_packages/funman_dreal

