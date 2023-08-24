#!/usr/bin/env bash

#TARGET_IMAGE=ghcr.io/jgladwig/funman-api:1.6.0
TARGET_IMAGE=localhost:5000/siftech/funman-api:local
docker run --rm -p 127.0.0.1:8190:8190 --pull always $TARGET_IMAGE