<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# CUDA AFD runtime sources

English | [简体中文](afd-runtime-sources_zh.md)

This record identifies source inputs for the synchronous CUDA
`P2pNcclAFDConnector` candidate. It does not establish successful image, GPU,
shutdown, or Kubernetes validation. DeepEPv2 GIN is outside this candidate.

## Fixed source inputs

| Input | Source value | Authority |
| --- | --- | --- |
| AFD plugin | `b7561d726b92f366f92634773b731b38e364fbe6` | [Package definition][plugin-package] |
| Plugin documentation snapshot | `913c9d35453fdf6b20553929ced9d5af1da7c0a3` | [Pinned README][plugin-readme] |
| vLLM | `0.30.0`, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | [Release tag reference][vllm-tag], [commit][vllm-commit] |
| CUDA toolkit | `13.0.3` | [vLLM Dockerfile][vllm-dockerfile] |
| Python / base distribution | `3.12` / Ubuntu `24.04` | [vLLM Dockerfile][vllm-dockerfile] |
| PyTorch | `2.13.0`, CUDA wheel `2.13.0+cu130` | [vLLM CUDA requirements][vllm-cuda], [CUDA wheel metadata][torch-metadata] |
| NCCL selected by vLLM image source | `nvidia-nccl-cu13==2.30.7` | [vLLM Dockerfile][vllm-dockerfile] |
| NCCL required by PyTorch wheel metadata | `nvidia-nccl-cu13==2.29.7` on Linux | [CUDA wheel metadata][torch-metadata] |
| NCCL selected for the Foretoken P2P candidate | `nvidia-nccl-cu13==2.29.7` | Matches the exact PyTorch requirement above |

The plugin requires Python `>=3.10,<3.14`; its optional runtime extra pins
`vllm==0.30.0`. The distribution is `vllm-afd-plugin`, the import is `afd_plugin`,
and its `vllm.general_plugins` entry point is `afd = afd_plugin:register_afd`.
The base package declares no runtime dependencies; `--no-deps` preserves the
selected environment without validating it. [Source: package definition][plugin-package].

The plugin uses `setuptools-scm`; a Git checkout preserves its version source.
An archive without that metadata must not be assumed to build identically.
`AFD_BUILD_ASCEND_OPS=0` disables the optional Ascend extension for CUDA builds.
[Sources: package definition][plugin-package], [build implementation][plugin-setup].

## NCCL selection

The fixed vLLM Dockerfile selects NCCL 2.30.7 for DeepEPv2 GIN, which needs
NCCL 2.30.4 or newer. The CPython 3.12 Linux amd64 CUDA 13.0 PyTorch wheel
instead requires exactly `nvidia-nccl-cu13==2.29.7`, as does its PyPI metadata.
[Sources: vLLM Dockerfile][vllm-dockerfile], [CUDA wheel metadata][torch-metadata],
[PyPI metadata][torch-pypi].

The AFD image therefore selects NCCL 2.29.7 to satisfy the pinned PyTorch
dependency and retains the normal dependency check. The plugin's synchronous
CUDA P2P connector uses vLLM's `PyNcclCommunicator` and does not require the
DeepEPv2 GIN backend. This choice is limited to the AFD image; it does not
change the ordinary CUDA, MetaX, or Omni runtime.
[Sources: CUDA wheel metadata][torch-metadata], [CUDA connector][plugin-p2p].
The upstream image persists its NCCL selection in `/etc/uv-overrides.txt`,
referenced by `UV_OVERRIDE`. The derived image must align that selection with
its installed NCCL version so later package installations retain the candidate.
[Source: vLLM Dockerfile][vllm-dockerfile].

## Native runtime boundary

The pinned README's 1A1F commands select `VLLM_USE_V2_MODEL_RUNNER=0`, DP=TP=1,
`--enable-expert-parallel`, `--enforce-eager`, and `P2pNcclAFDConnector`. Both
roles use the same rendezvous and rank counts; requests go to Attention.
`async_dp` and `compute_gate_on_attention` default to false.
[Sources: pinned README][plugin-readme], [configuration][plugin-config].

The connector uses a collective TCP rendezvous and orders FFN ranks before
Attention ranks. Participants must agree on endpoint, rank counts, model shape,
dtype, and role assignment. Same-node placement does not prove cross-Pod GPU
communication; cleanup code does not prove process termination or GPU resource
release. [Source: CUDA connector][plugin-p2p].

## Evidence still required

The official release lists `vllm/vllm-openai:v0.30.0` for CUDA 13.0 and the
separate `v0.30.0-cu129` variant for CUDA 12.9. Tags are not immutable pins.
[Source: vLLM release][vllm-release].

- An immutable `linux/amd64` image digest and measured Python, vLLM, PyTorch,
  CUDA, and NCCL versions; no OCI digest is established here.
- One fixed [DeepSeek-V2-Lite repository][model-repository] revision for weights
  and tokenizer; no model revision is established here.
- A successful Foretoken image build and dependency check in the actual
  `FORETOKEN_VLLM_PYTHON` environment.
- Owner-executed A100 1A1F generation, ordered shutdown, and same-node cross-Pod
  communication, including request output, exit status, remaining GPU processes,
  and Pod state. Imports and open ports are insufficient acceptance.

[plugin-package]: https://github.com/vllm-project/afd-plugin/blob/b7561d726b92f366f92634773b731b38e364fbe6/pyproject.toml
[plugin-readme]: https://github.com/vllm-project/afd-plugin/blob/913c9d35453fdf6b20553929ced9d5af1da7c0a3/README.md
[plugin-setup]: https://github.com/vllm-project/afd-plugin/blob/b7561d726b92f366f92634773b731b38e364fbe6/setup.py
[plugin-config]: https://github.com/vllm-project/afd-plugin/blob/b7561d726b92f366f92634773b731b38e364fbe6/afd_plugin/config.py
[plugin-p2p]: https://github.com/vllm-project/afd-plugin/blob/b7561d726b92f366f92634773b731b38e364fbe6/afd_plugin/connectors/gpu/p2p.py
[vllm-tag]: https://api.github.com/repos/vllm-project/vllm/git/ref/tags/v0.30.0
[vllm-release]: https://github.com/vllm-project/vllm/releases/tag/v0.30.0
[vllm-commit]: https://github.com/vllm-project/vllm/commit/ced6857afa0ea7b2e3f0846a62e1394e90f15607
[vllm-dockerfile]: https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/docker/Dockerfile
[vllm-cuda]: https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/requirements/cuda.txt
[torch-metadata]: https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl.metadata
[torch-pypi]: https://pypi.org/pypi/torch/2.13.0/json
[model-repository]: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
