ARG SIFT_REGISTRY_ROOT
ARG FROM_IMAGE=funman-pypi
ARG FROM_TAG=${TARGETOS}-${TARGETARCH}
FROM ${SIFT_REGISTRY_ROOT}${FROM_IMAGE}:${FROM_TAG}

RUN pip install --no-cache-dir uvicorn

EXPOSE 8190
CMD [ "uvicorn", "funman.api.api:app", "--host", "0.0.0.0", "--port", "8190" ]

