<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# Build and verify the CUDA AFD image

English | [简体中文](afd-runtime_zh.md)

Build an explicitly selected CUDA environment for native Attention/FFN
Disaggregation (AFD) validation. This image supplies the plugin; it does not
add Foretoken AFD launch plans, scheduling, or readiness. A100 and K3s
acceptance remain owner-executed work before this candidate is supported.

## Build on Linux amd64

Use Docker with BuildKit and Make from the repository root. Resolve the
official CUDA 13.0 release image to a digest on the build server and retain it:

```bash
docker pull --platform linux/amd64 vllm/vllm-openai:v0.30.0
AFD_BASE_IMAGE=$(docker image inspect vllm/vllm-openai:v0.30.0 \
  --format '{{index .RepoDigests 0}}')
AFD_IMAGE=foretoken-model-server-afd:$(git rev-parse --short=12 HEAD)
make image-model-server-afd \
  VLLM_AFD_BASE_IMAGE="$AFD_BASE_IMAGE" \
  AFD_MODEL_SERVER_IMAGE="$AFD_IMAGE"
```

The target builds `foretoken-vllm-afd:dev`, then wraps it through the existing
model-server build; both target `linux/amd64`. Override `VLLM_AFD_IMAGE` for
another intermediate name. `IMAGE_TARGET=environment` and `IMAGE_OUTPUT`
retain their existing meaning for the final build. Ordinary model-server,
MetaX, and Omni builds do not select AFD. Platform defaults are unchanged.

The AFD targets select `/usr/bin/python3` by default; an explicit
`FORETOKEN_VLLM_PYTHON` selects another interpreter for both builds. The plugin
wheel is built separately and installed in that environment. The build checks vLLM 0.30.0,
PyTorch 2.13.0/cu130, CUDA 13.0.3, and NCCL 2.29.7, runs `uv pip check`,
and writes measured versions to `/usr/share/foretoken/afd-runtime.json`.
The NCCL selection also replaces the inherited uv override for subsequent
package installations. This synchronous P2P candidate does not support
DeepEPv2 GIN. See the [fixed sources](afd-runtime-sources.md).

The Rust workspace retains its separate vLLM gitlink. `VLLM_REVISION` controls
that build source, not the Python engine; do not upgrade it for this image.

## Inspect the artifact

```bash
docker image inspect "$AFD_IMAGE" \
  --format 'id={{.Id}} platform={{.Os}}/{{.Architecture}} labels={{json .Config.Labels}}'
docker run --rm --entrypoint cat "$AFD_IMAGE" \
  /usr/share/foretoken/afd-runtime.json
docker run --rm --entrypoint sh "$AFD_IMAGE" -c \
  '"${FORETOKEN_VLLM_PYTHON:-python}" -m pip check'
```

Expect the fixed plugin revision, the versions above, the intended Python
executable, and no dependency errors. These are build checks, not inference
acceptance. Record the final OCI manifest digest separately from the local
image ID, which is a configuration digest. Retain source SHA, input reference,
build command and log with the artifact.

## Owner: native 1A1F on A100

Use two A100 80GB GPUs, a CUDA 13.0-compatible NVIDIA driver, NVIDIA Container
Toolkit, and the built image. Record one full commit from the official
`deepseek-ai/DeepSeek-V2-Lite` repository for both model and tokenizer. No
model revision is established by this image. Set `MODEL_CACHE` to an existing
server cache directory. Export both values in the two host terminals:

```bash
export MODEL_REVISION='<full DeepSeek-V2-Lite commit>'
export MODEL_CACHE='<existing absolute model-cache directory>'
```

Start two containers from the same image, each with one distinct GPU and
`--network host --ipc host`, mounting the model cache. Override the image
entrypoint with `sh`; inside each container run the following command with
`ROLE=ffn` / `HTTP_PORT=18001` or `ROLE=attention` / `HTTP_PORT=18000`. Both
processes must start to complete the shared rendezvous. In the first terminal:

```bash
docker run -it --name foretoken-afd-ffn --gpus device=1 \
  --network host --ipc host -v "$MODEL_CACHE:/root/.cache/huggingface" \
  -e MODEL_REVISION -e ROLE=ffn -e HTTP_PORT=18001 \
  --entrypoint sh "$AFD_IMAGE"
```

In the second terminal:

```bash
docker run -it --name foretoken-afd-attention --gpus device=0 \
  --network host --ipc host -v "$MODEL_CACHE:/root/.cache/huggingface" \
  -e MODEL_REVISION -e ROLE=attention -e HTTP_PORT=18000 \
  --entrypoint sh "$AFD_IMAGE"
```

Run this command inside each container:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
exec "${FORETOKEN_VLLM_PYTHON:-python}" -m vllm.entrypoints.cli.main serve \
  deepseek-ai/DeepSeek-V2-Lite --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" --trust-remote-code \
  --served-model-name deepseek-v2-lite-afd --dtype bfloat16 \
  --max-model-len 2048 --data-parallel-size 1 --tensor-parallel-size 1 \
  --enable-expert-parallel --enforce-eager --host 127.0.0.1 --port "$HTTP_PORT" \
  --additional-config "{\"afd\":{\"role\":\"$ROLE\",\"connector\":\"P2pNcclAFDConnector\",\"host\":\"127.0.0.1\",\"port\":6239,\"num_attention_ranks\":1,\"num_ffn_ranks\":1,\"async_dp\":false,\"compute_gate_on_attention\":false}}"
```

Keep PP=PCP=DCP=1. After initialization, send a request to Attention:

```bash
curl --fail-with-body http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v2-lite-afd","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Expect nonempty generated text without a collective failure or stalled
request. Imports and open ports are insufficient. After requests complete,
stop Attention before FFN. Retain logs and actual exit codes; check that
neither process requires a forced kill and `nvidia-smi` shows no orphaned
engine or retained GPU allocation. Run from a third host terminal:

```bash
docker stop --time 60 foretoken-afd-attention
docker stop --time 60 foretoken-afd-ffn
docker inspect foretoken-afd-attention foretoken-afd-ffn --format '{{json .State}}'
docker logs foretoken-afd-attention
docker logs foretoken-afd-ffn
nvidia-smi
docker rm foretoken-afd-attention foretoken-afd-ffn
```

## Owner: same-node cross-Pod communication

Repeat the native commands in two temporary Pods on the same A100 node,
each requesting one GPU, using the same image digest and fixed revisions.
Record the K3s server version, Pod specifications, selected GPUs, network
interfaces, and NCCL environment. Both roles use one rendezvous address
reachable at the FFN Pod; `127.0.0.1` only works for a shared network
namespace. Expose Attention's HTTP listener in the validation network.

Start both roles concurrently and repeat generation and ordered shutdown.
Expect actual tensor exchange across Pod boundaries and released resources.
Save request output, communication logs, container/Pod exit statuses, and GPU
allocations after termination. Host-network success alone does not prove this
case. This is native validation; Foretoken-managed AFD lifecycle and
controller deployment are outside the image's scope.
