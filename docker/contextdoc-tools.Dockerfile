FROM python:3.11-slim

ARG CONTEXTDOC_REF=main

RUN apt-get update \
  && apt-get install -y --no-install-recommends git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${CONTEXTDOC_REF} \
    https://github.com/MatteoAdamo82/contextdoc.git /opt/contextdoc

RUN pip install --no-cache-dir \
    -r /opt/contextdoc/tools/ctx-run/requirements.txt \
    -r /opt/contextdoc/tools/ctx-watch/requirements.txt

WORKDIR /workspace
