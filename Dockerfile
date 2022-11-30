FROM ubuntu:20.04

# Install python
RUN apt update && apt install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install automates dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends \
    antlr4 \
    build-essential \
    gcc \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Initialize python virtual environment
ENV VIRTUAL_ENV=/funman_dev
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade setuptools
RUN pip install --no-cache-dir wheel

RUN pip install --no-cache-dir pytest==7.1.2
RUN pip install --no-cache-dir sphinx==5.2.2
RUN pip install --no-cache-dir sphinx-rtd-theme==1.0.0
RUN pip install --no-cache-dir twine
RUN pip install --no-cache-dir build

# Install automates
RUN apt update && apt install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth=1 https://github.com/danbryce/automates.git /automates \
    && cd /automates \
    && git fetch --depth=1 origin e5fb635757aa57007615a75371f55dd4a24851e0 \
    && git checkout e5fb635757aa57007615a75371f55dd4a24851e0
RUN pip install -e /automates
# RUN pip install --no-cache-dir https://github.com/ml4ai/automates/archive/v1.3.0.zip
# RUN pip install https://github.com/danbryce/automates/archive/e5fb635757aa57007615a75371f55dd4a24851e0.zip

RUN pip install --no-cache-dir z3-solver

# Install funman dev packages
COPY ./model2smtlib model2smtlib
RUN pip install -e /model2smtlib
# RUN --mount=type=bind,source=./model2smtlib,target=/model2smtlib,rw \
#     pip install -e /model2smtlib

COPY ./funman funman
RUN pip install -e /funman
# RUN --mount=type=bind,source=./funman,target=/funman,rw \
#     pip install -e /funman

RUN pip install -e /funman/auxiliary_packages/funman_demo
# RUN --mount=type=bind,source=./funman,target=/funman,rw \
#     pip install -e /funman/auxiliary_packages/funman_demo

WORKDIR /funman
CMD /bin/bash