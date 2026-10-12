# syntax=docker/dockerfile:1.19
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

# Build from the repository root so Cargo uses the data-plane workspace and
# its engine adapters. The inference engine image must provide the Python
# entrypoint required by the selected adapter. FORETOKEN_VLLM_PYTHON may select
# an image-specific executable; model-server otherwise uses `python`. The
# current vLLM adapter uses vllm.entrypoints.cli.main.

ARG BASE_IMAGE_REGISTRY=docker.io
ARG UV_IMAGE_REGISTRY=ghcr.io
ARG INFERENCE_ENGINE_IMAGE_REGISTRY=ghcr.io
ARG RUST_BUILDER_IMAGE=${BASE_IMAGE_REGISTRY}/library/rust:1.97.1-bullseye
ARG INFERENCE_ENGINE_IMAGE=${INFERENCE_ENGINE_IMAGE_REGISTRY}/shiweijiezero/foretoken/model-server
ARG VLLM_AFD_BASE_IMAGE=vllm/vllm-openai:v0.30.0
ARG FORETOKEN_VLLM_PYTHON
ARG UV_IMAGE=${UV_IMAGE_REGISTRY}/astral-sh/uv:0.9.10
ARG AFD_PLUGIN_REVISION=b7561d726b92f366f92634773b731b38e364fbe6

FROM ${UV_IMAGE} AS uv

# Keep the build-time patch utility out of the inference image.
FROM ${BASE_IMAGE_REGISTRY}/library/debian:bookworm-slim AS patch-tool
RUN apt-get update \
    && apt-get install -y --no-install-recommends patch ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Build the Python-only CUDA plugin with Git metadata for setuptools-scm.
# Keep its packaging tools out of the engine image.
FROM patch-tool AS afd-plugin-build
ARG AFD_PLUGIN_REVISION
ARG FORETOKEN_GITHUB_MIRROR
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
RUN apt-get update \
    && apt-get install -y --no-install-recommends git python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN origin="${FORETOKEN_GITHUB_MIRROR:-https://github.com}" \
    && git init /src/afd-plugin \
    && git -C /src/afd-plugin fetch --depth=1 \
      "${origin%/}/vllm-project/afd-plugin.git" "${AFD_PLUGIN_REVISION}" \
    && git -C /src/afd-plugin checkout --detach FETCH_HEAD
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    AFD_BUILD_ASCEND_OPS=0 uv build --python python3 --wheel \
      --default-index "${UV_DEFAULT_INDEX}" --out-dir /wheels /src/afd-plugin

# The upstream musl binary also runs in inference images with older glibc.
FROM patch-tool AS dragonfly-client
ARG TARGETARCH
ARG DRAGONFLY_CLIENT_VERSION=1.5.5
ARG FORETOKEN_GITHUB_MIRROR
RUN case "$TARGETARCH" in amd64) arch=x86_64 ;; arm64) arch=aarch64 ;; *) exit 1 ;; esac \
    && package="dragonfly-client-v${DRAGONFLY_CLIENT_VERSION}-${arch}-unknown-linux-musl" \
    && origin="${FORETOKEN_GITHUB_MIRROR:-https://github.com}" \
    && curl --fail --location "${origin%/}/dragonflyoss/client/releases/download/v${DRAGONFLY_CLIENT_VERSION}/${package}.tar.gz" --output /tmp/client.tar.gz \
    && mkdir /out \
    && tar -xzf /tmp/client.tar.gz --strip-components=1 -C /out "$package/dfget" "$package/LICENSE" \
    && chmod 755 /out/dfget

FROM ${RUST_BUILDER_IMAGE} AS builder

ARG FORETOKEN_GITHUB_MIRROR
ARG FORETOKEN_CARGO_REGISTRY
ARG CARGO_NET_GIT_FETCH_WITH_CLI
ARG FORETOKEN_MODEL_SERVER_BINARY=foretoken-model-server
ARG VLLM_REVISION

WORKDIR /workspace

# Python adapters belong to the file export, not the Rust compilation input.
COPY --exclude=model-server/python data-plane/ ./data-plane/

# Bullseye cannot load procedural macros linked against the frontend builder's newer glibc.
# Retain prepared Rust inputs with their outputs, isolated from legacy builders.
# Copy changed trees afresh so patch removals also receive mtimes Cargo can observe.
RUN --mount=type=cache,id=foretoken-vllm-${VLLM_REVISION},target=/var/cache/vllm,sharing=locked \
    --mount=type=cache,id=foretoken-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=foretoken-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=foretoken-cargo-target-model-server-prepared-source,target=/workspace/data-plane/target,sharing=locked \
    if [ -n "${FORETOKEN_GITHUB_MIRROR}" ]; then \
      printf '[url "%s/"]\n\tinsteadOf = https://github.com/\n' \
        "${FORETOKEN_GITHUB_MIRROR%/}" > /root/.gitconfig; \
    fi \
    && if [ -n "${FORETOKEN_CARGO_REGISTRY}" ]; then \
      mkdir -p /workspace/.cargo; \
      printf '[source.crates-io]\nreplace-with = "foretoken-mirror"\n\n[source.foretoken-mirror]\nregistry = "%s"\n' \
        "${FORETOKEN_CARGO_REGISTRY}" > /workspace/.cargo/config.toml; \
    fi \
    && if [ -n "${CARGO_NET_GIT_FETCH_WITH_CLI}" ]; then \
      export CARGO_NET_GIT_FETCH_WITH_CLI; \
    fi \
    && if [ ! -f data-plane/third_party/vllm/vllm/v1/engine/core.py ]; then \
      if [ ! -f /var/cache/vllm/rust/Cargo.toml ]; then \
        git -C /var/cache/vllm init -q \
        && git -C /var/cache/vllm fetch --depth=1 https://github.com/vllm-project/vllm.git "${VLLM_REVISION}" \
        && git -C /var/cache/vllm checkout -q --detach FETCH_HEAD; \
      fi \
      && mkdir -p data-plane/third_party/vllm \
      && git -C /var/cache/vllm archive HEAD | tar -x -C data-plane/third_party/vllm; \
    fi \
    && cargo run --locked --manifest-path data-plane/xtask/Cargo.toml -- prepare-vllm \
    && mkdir -p data-plane/target/vllm-rust \
    && if git diff --no-index --quiet data-plane/target/vllm-rust data-plane/third_party/vllm/rust; then \
      :; \
    else \
      result=$?; \
      test "$result" -eq 1 \
      && rm -rf data-plane/target/vllm-rust \
      && cp -R data-plane/third_party/vllm/rust data-plane/target/vllm-rust; \
    fi \
    && rm -rf data-plane/third_party/vllm/rust \
    && cp -a data-plane/target/vllm-rust data-plane/third_party/vllm/rust \
    && cargo build --locked --release \
      --manifest-path data-plane/Cargo.toml \
      --package foretoken-model-server --bin "${FORETOKEN_MODEL_SERVER_BINARY}" \
    && install -Dm755 \
      "data-plane/target/release/${FORETOKEN_MODEL_SERVER_BINARY}" \
      /out/model-server

FROM scratch AS source-export
COPY --from=builder /out/model-server /bin/foretoken-model-server
COPY data-plane/model-server/python/ /python/
COPY LICENSE /LICENSE
COPY <<EOF /complete.json
{"component":"model-server"}
EOF

# vLLM-Omni images already contain their Python runtime and model plugins. This
# target only adds the thin Foretoken lifecycle/proxy adapter and deliberately
# skips the standard vLLM patch and dependency stages below.
FROM ${INFERENCE_ENGINE_IMAGE} AS omni-runtime

ARG OMNI_VIDEO_SYNC_TIMEOUT=4000
ENV VLLM_OMNI_VIDEO_SYNC_TIMEOUT=${OMNI_VIDEO_SYNC_TIMEOUT} \
    VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=${OMNI_VIDEO_SYNC_TIMEOUT}

ARG OCI_SOURCE=https://github.com/shiweijiezero/foretoken
ARG OCI_REVISION
LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.revision="${OCI_REVISION}"
COPY --from=builder /out/model-server /usr/local/bin/foretoken-omni-model-server
ENTRYPOINT ["foretoken-omni-model-server"]

# The explicit CUDA AFD base owns its plugin and dependency alignment. The
# existing engine-runtime stages then wrap it into the model-server image.
FROM ${VLLM_AFD_BASE_IMAGE} AS afd-engine
ARG FORETOKEN_VLLM_PYTHON
ENV FORETOKEN_VLLM_PYTHON=${FORETOKEN_VLLM_PYTHON} \
    UV_OVERRIDE=/etc/uv-overrides.txt
ARG AFD_PLUGIN_REVISION
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
ARG OCI_SOURCE=https://github.com/shiweijiezero/foretoken
ARG OCI_REVISION
LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.revision="${OCI_REVISION}" \
      io.foretoken.afd.plugin.revision="${AFD_PLUGIN_REVISION}"
# Upstream overrides NCCL for DeepEP v2. Synchronous AFD P2P uses PyTorch's
# declared NCCL dependency instead; this variant does not support DeepEP v2.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=from=afd-plugin-build,source=/wheels,target=/tmp/afd-plugin <<'SH'
set -eu
python_path=$("${FORETOKEN_VLLM_PYTHON:-python}" -c 'import sys; print(sys.executable)')
printf '%s\n' 'nvidia-nccl-cu13==2.29.7' > "$UV_OVERRIDE"
uv pip install --no-cache --no-deps --python "$python_path" \
    --default-index "${UV_DEFAULT_INDEX}" --requirements "$UV_OVERRIDE"
uv pip install --no-cache --no-deps --python "$python_path" /tmp/afd-plugin/*.whl
uv pip check --python "$python_path"
"$python_path" - <<'PY'
import json
import os
import platform
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit("CUDA AFD requires the Python 3.12 runtime")
expected = {
    "vllm": "0.30.0",
    "torch": "2.13.0",
    "nvidia-nccl-cu13": "2.29.7",
}
packages = {name: version(name) for name in expected}
for name, required in expected.items():
    if packages[name].split("+")[0] != required:
        raise SystemExit(f"CUDA AFD requires {name}=={required}, got {packages[name]}")
if torch.version.cuda != "13.0" or os.environ.get("CUDA_VERSION") != "13.0.2":
    raise SystemExit("CUDA AFD requires the CUDA 13.0.2 image and cu130 PyTorch")
plugin = distribution("vllm-afd-plugin")
packages[plugin.name] = plugin.version
record = {
    "python": sys.executable,
    "python_version": platform.python_version(),
    "packages": packages,
    "cuda": os.environ["CUDA_VERSION"],
    "torch_cuda": torch.version.cuda,
    "plugin_revision": os.environ["AFD_PLUGIN_REVISION"],
}
output = Path("/usr/share/foretoken/afd-runtime.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2) + "\n")
print(output.read_text())
PY
SH

FROM ${INFERENCE_ENGINE_IMAGE} AS engine-runtime

ARG FORETOKEN_VLLM_PYTHON
ENV FORETOKEN_VLLM_PYTHON=${FORETOKEN_VLLM_PYTHON}

# Patch the installed Python engine, not the independent Rust build source.
# A derived Foretoken image may already contain this exact backport.
RUN --mount=from=patch-tool,source=/usr/bin/patch,target=/usr/local/bin/patch \
    --mount=type=bind,source=data-plane/patches/vllm,target=/tmp/vllm-patches \
    "${FORETOKEN_VLLM_PYTHON:-python}" /tmp/vllm-patches/apply.py

ARG UV_DEFAULT_INDEX=https://pypi.org/simple
# Engine images may provide a Python virtual environment without pip.
# MACA also exposes torch.version.cuda, so select dependencies from the engine build.
# Precompile installed packages to avoid repeated bytecode compilation at startup.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    python_path=$("${FORETOKEN_VLLM_PYTHON:-python}" -c \
      'import sys; print(sys.executable)') \
    && backend=$("$python_path" -c \
      'import torch; print("maca" if getattr(torch.version, "maca", None) else "cuda" if torch.version.cuda else "other")') \
    && set -- modelscope==1.35.3 tzdata \
    && case "$backend" in cuda) set -- "$@" bitsandbytes==0.49.2 modelexpress==0.6.0 nixl==1.4.1 ;; esac \
    && "$python_path" -c 'from importlib.metadata import version; print("torch==" + version("torch")); print("vllm==" + version("vllm"))' > /tmp/foretoken-engine-constraints.txt \
    && uv pip install --no-cache --python "$python_path" \
      --constraint /tmp/foretoken-engine-constraints.txt \
      --default-index "${UV_DEFAULT_INDEX}" "$@" \
    && uv pip check --python "$python_path" \
    && "$python_path" -c 'import compileall, site; raise SystemExit(not all(compileall.compile_dir(path, quiet=1) for path in site.getsitepackages()))' \
    && rm /tmp/foretoken-engine-constraints.txt

FROM engine-runtime AS environment
ARG OCI_SOURCE=https://github.com/shiweijiezero/foretoken
ARG OCI_REVISION
LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.revision="${OCI_REVISION}"
COPY --from=dragonfly-client /out/dfget /usr/local/bin/dfget
COPY --from=dragonfly-client /out/LICENSE /usr/share/licenses/dragonfly-client/LICENSE

FROM environment AS runtime
COPY --from=builder /out/model-server /usr/local/bin/foretoken-model-server
COPY data-plane/model-server/python/ /opt/foretoken/python/

# Preserve the inference engine image's runtime user and environment. The workload owns its
# security context, cache environment, and writable volume mounts.
ENTRYPOINT ["foretoken-model-server"]
