<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# Deploy Foretoken from Source

[English](custom-deployment.md) | [中文](custom-deployment_zh.md)

Build Foretoken from a local checkout and deploy source changes to Kubernetes.

## Install from source

Prepare Python 3.11+, Git, kubectl, and Helm. The cluster must allow BuildKit Pods and publishing Jobs, with a default StorageClass for compiler caches and published application files. To override their storage classes, set `development.build.storageClassName` or `applicationFiles.storageClassName` in `deploy/platform-values.yaml` and pass it with `--values`.

```bash
git clone https://github.com/shiweijiezero/foretoken.git
cd foretoken
kubectl config current-context
```

Install the CLI from this checkout:

```bash
pip install -e .
```

For a local kind or k3d cluster, build and install without a registry:

```bash
foretoken install -e .
```

Builds run in the cluster. For GPU setup in k3d, see [Deploy Foretoken with k3d](k3d-deployment.md).

## Deploy and update code

Deploy the maintained [Quick Start](../README.md#quick-start) on a GPU-enabled cluster:

```bash
foretoken deploy examples/quickstart --timeout 20m
```

After editing the checkout, run the same command again. It uses the saved installation settings and reuses the runtime environment for code updates. Changes to runtime dependencies or image build settings update the platform installation.

Affected workloads restart and may reload model weights. The command waits for the selected code and serving routes to become active. Unchanged source and deployment configuration leave existing workloads running. Use the Quick Start's [request](../README.md#4-send-a-test-request) and [cleanup](../README.md#stop-and-uninstall) commands.

Changes to CLI Python files take effect directly from the editable checkout; rerun `pip install -e .` when its Python dependencies change.

## CUDA AFD source deployment

On NVIDIA, the same `foretoken install -e .` command also builds the fixed CUDA AFD environment. Pools with `afd` select it automatically; ordinary CUDA, MetaX and Omni keep their existing environments. Use the [AFD Pool fields](inference-parameters.md#attentionffn-pool-configuration) with `foretoken deploy`; no private launch plan is required. This is candidate delivery, not a validated GPU support matrix.

Rust and the Python observation adapter come from this checkout. AFD uses its own immutable application directory and the [fixed plugin/vLLM environment](development/afd-runtime-sources.md). `--engine-source` applies only to ordinary vLLM/MetaX; its Python and native overlays are never added to AFD. AFD plugin or Python engine changes require a compatible environment rebuild.

For a reproducible server build, set `VLLM_AFD_BASE_IMAGE` to the recorded upstream CUDA image digest before installation. The Dockerfile's versioned tag remains a candidate until that digest is recorded. An explicit `FORETOKEN_VLLM_PYTHON` must identify the compatible interpreter in every selected build environment.

`runtime.vllm.afd.image` defaults to `auto`; in source mode this builds the repository's AFD stage on NVIDIA. An explicit image supplies an already compatible AFD engine environment; an empty string disables AFD delivery. Save such overrides in the same values file and rerun the existing install/deploy commands. The source binding retains the input base separately from generated image references.

Code updates publish ordinary and AFD applications at the same source revision before a mixed ModelService selects them. Image dependency changes use the existing installation update. See [version retention and rollback](development/source-image-lifecycle.md#afd-environment-and-application-selection) before changing an existing AFD deployment.

## Edit an inference engine

To modify vLLM, bind a Git checkout matching the runtime's Python, PyTorch, and accelerator environment. For a checkout at `../vllm`:

```bash
foretoken install -e . --engine-source ../vllm
```

Retain `--registry` and `--values` when using them. After editing Python, Triton or NVIDIA CUDA/C++ source, use `foretoken deploy` as above; required compilation is automatic.

For MetaX, native kernels belong to the plugin checkout. Bind it alongside the matching core checkout:

```bash
foretoken install -e . \
  --engine-source ../vllm \
  --engine-source vllm-metax=../vllm-metax
```

### Select a different runtime environment

The runtime image supplies Python, PyTorch, and accelerator libraries. To change that environment, set a compatible image in `deploy/platform-values.yaml`, replacing the example with an image available to the cluster builder:

```yaml
runtime:
  vllm:
    image: ghcr.io/example/custom-vllm:latest
```

Reapply the installation command with `--values deploy/platform-values.yaml`, retaining the registry and engine-source options. With `-e`, Foretoken uses this image as its build base. Then deploy the workload again.

MetaX base-image builds are covered by [Prepare Foretoken for MetaX GPUs](development/metax-platform.md#install-from-source).

### vLLM-Omni runtime

vLLM-Omni uses a separate model-server image. Build it from the repository root on a machine with Docker BuildKit, Make, and a rustup-managed Rust toolchain:

```bash
make image-vllm-omni VLLM_OMNI_IMAGE=foretoken-vllm-omni:latest
make image-model-server-omni \
  INFERENCE_ENGINE_IMAGE=foretoken-vllm-omni:latest \
  OMNI_MODEL_SERVER_IMAGE=foretoken-omni-model-server:latest
```

For k3d, set `CLUSTER` to the existing cluster name and import the image:

```bash
CLUSTER=foretoken-qwen-test
k3d image import --cluster "$CLUSTER" foretoken-omni-model-server:latest
```

For a remote cluster, use the `REGISTRY` configured in the [Kubernetes deployment guide](kubernetes-deployment.md). Log in first when that registry requires authentication:

```bash
docker tag foretoken-omni-model-server:latest "$REGISTRY/omni-model-server:latest"
docker push "$REGISTRY/omni-model-server:latest"
```

Set `runtime.vllmOmni.image` in `deploy/platform-values.yaml` to the image the nodes can pull. This example uses the local image; for a registry, replace it with the full pushed reference:

```yaml
runtime:
  vllmOmni:
    image: foretoken-omni-model-server:latest
```

Apply the runtime setting:

```bash
foretoken install -e . --values deploy/platform-values.yaml
```

Retain `--registry "$REGISTRY"` for remote platform builds and any other installation options. The editable vLLM source path above targets the standard vLLM backend.

After installation, run `foretoken deploy` with the Kustomize directory for your Omni service. For code updates, rebuild and distribute the Omni image, then set `runtime.vllmOmni.image` to a new tag or digest reference. Reapply the installation command and deploy the same Kustomize directory again.
