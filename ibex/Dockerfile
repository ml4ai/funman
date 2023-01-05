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
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Initialize python virtual environment
ENV VIRTUAL_ENV="/.ibex_env"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# COPY ./ibex-lib /ibex-lib
RUN git clone \
  --depth 1 \
  --branch ibex-2.8.9_using_mathlib-2.1.1 \
  https://github.com/jgladwig/ibex-lib.git

COPY ./Makefile.ibex /ibex-lib/Makefile

ARG TARGETOS
ENV TARGETOS=$TARGETOS

ARG TARGETARCH
ENV TARGETARCH=$TARGETARCH

ARG TARGETPLATFORM
ENV TARGETPLATFOMR=$TARGETPLATFORM

WORKDIR /ibex-lib
RUN make mathlib
RUN make gaol
RUN make ibex

CMD [ "/bin/bash" ]
