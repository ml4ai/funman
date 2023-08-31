#!/usr/bin/env bash

# For local images:
# TARGET_IMAGE=localhost:5000/siftech/funman-api:local

# For public images:
FUNMAN_VERSION="1.6.0"
TARGET_IMAGE=ghcr.io/jgladwig/funman-api:$FUNMAN_VERSION

docker run --rm -p 127.0.0.1:8190:8190 --pull always $TARGET_IMAGE