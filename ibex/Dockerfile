FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
# Install base dependencies
RUN apt update && apt install -y --no-install-recommends \
    autoconf \
    automake \
    bison \
    cmake \
    curl \
    flex \
    g++ \
    gcc \
    git \
    libtool \
    m4 \
    make \
    ninja-build \
    patch \
    pkg-config \
    python-is-python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# COPY ./ibex-lib /ibex-lib
RUN git clone \
  --depth 1 \
  --branch ibex-2.8.5_using_mathlib-2.1.1 \
  https://github.com/jgladwig/ibex-lib.git

COPY ./Makefile.ibex /ibex-lib/Makefile

ENV TARGETOS=$TARGETOS
ENV TARGETARCH=$TARGETARCH
ENV TARGETPLATFORM=$TARGETPLATFORM

WORKDIR /ibex-lib
RUN make ibex INSTALL_PREFIX=/usr LP_LIB=clp

CMD [ "/bin/bash" ]
