ARG SIFT_REGISTRY_ROOT
ARG DREAL_TAG=${TARGETOS}-${TARGETARCH}
FROM ${SIFT_REGISTRY_ROOT}funman-dreal4:${DREAL_TAG}

# Install base dependencies
RUN apt update && apt install -y --no-install-recommends \
    curl \
    git \
    make \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install automates dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    antlr4 \
    build-essential \
    gcc \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# setup non-root user
ARG UNAME=funman
ARG UID=1000
ARG GID=1000
RUN groupadd -o -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -G sudo -s /bin/bash $UNAME
USER $UNAME

# Initialize python virtual environment
ENV VIRTUAL_ENV="/home/$UNAME/funman_venv"
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PYTHONPYCACHEPREFIX="/home/$UNAME/.cache/pycache/"

RUN pip install --no-cache-dir --upgrade setuptools pip
RUN pip install --no-cache-dir wheel

RUN pip install --no-cache-dir pytest==7.1.2
RUN pip install --no-cache-dir sphinx==5.2.2
RUN pip install --no-cache-dir sphinx-rtd-theme==1.0.0
RUN pip install --no-cache-dir twine
RUN pip install --no-cache-dir build
RUN pip install --no-cache-dir pylint
RUN pip install --no-cache-dir black
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir openapi-generator
RUN pip install --no-cache-dir pre-commit
RUN pip install --no-cache-dir pycln
RUN pip install --no-cache-dir openapi-python-client

WORKDIR /home/$UNAME

# Install automates
RUN git clone --depth=1 https://github.com/danbryce/automates.git automates \
    && cd automates \
    && git fetch --depth=1 origin e5fb635757aa57007615a75371f55dd4a24851e0 \
    && git checkout e5fb635757aa57007615a75371f55dd4a24851e0
RUN pip install -e automates

RUN pip install --no-cache-dir z3-solver
RUN pip install --no-cache-dir graphviz
RUN pip install --no-cache-dir \
    jupyterlab \
    jupyterlab-vim

RUN pip install /dreal4/dreal-*.whl

# Install funman dev packages
COPY --chown=$UID:$GID . funman
RUN pip install -e funman
RUN pip install -e funman/auxiliary_packages/funman_demo
RUN pip install -e funman/auxiliary_packages/funman_dreal

# configure a script for updating the local version of dreal
RUN mkdir -p /home/$UNAME/.local/bin
RUN echo 'PATH="$HOME/.local/bin:$PATH"' >> /home/$UNAME/.bashrc
COPY --chmod=755 tools/update-dreal.user /home/$UNAME/.local/bin/update-dreal
COPY --chmod=755 tools/funman-notebook.sh /home/$UNAME/.local/bin/funman-notebook
COPY --chmod=755 tools/funman-lab.sh /home/$UNAME/.local/bin/funman-lab
COPY --chmod=755 tools/funman-api-server.sh /home/$UNAME/.local/bin/funman-api-server
USER root
COPY --chmod=744 tools/update-dreal.root /usr/local/bin/update-dreal
RUN echo "$UNAME ALL=(ALL) NOPASSWD:/usr/local/bin/update-dreal" >> /etc/sudoers
RUN echo "%$UNAME ALL=(ALL) NOPASSWD:/usr/local/bin/update-dreal" >> /etc/sudoers

# for debugging
RUN apt update && apt install -y --no-install-recommends \
    vim \
    && rm -rf /var/lib/apt/lists/*
USER $UNAME

WORKDIR /home/$UNAME/funman
CMD /bin/bash
