<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# CUDA AFD 运行环境的版本来源

[English](afd-runtime-sources.md) | 简体中文

本文记录同步 CUDA `P2pNcclAFDConnector` 候选环境的源码依据，不代表镜像、
GPU 推理、退出或 Kubernetes 验收通过。DeepEPv2 GIN 不在该候选范围内。

## 固定输入

| 输入 | 版本 | 权威来源 |
| --- | --- | --- |
| AFD 插件 | `b7561d726b92f366f92634773b731b38e364fbe6` | [包定义][plugin-package] |
| 插件文档快照 | `913c9d35453fdf6b20553929ced9d5af1da7c0a3` | [固定 README][plugin-readme] |
| vLLM | `0.30.0`，commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | [tag][vllm-tag]、[commit][vllm-commit] |
| CUDA toolkit | `13.0.3` | [Dockerfile][vllm-dockerfile] |
| Python / 系统 | `3.12` / Ubuntu `24.04` | [Dockerfile][vllm-dockerfile] |
| PyTorch | `2.13.0`，CUDA wheel 为 `2.13.0+cu130` | [依赖][vllm-cuda]、[wheel 元数据][torch-metadata] |
| vLLM 镜像源码选择的 NCCL | `nvidia-nccl-cu13==2.30.7` | [Dockerfile][vllm-dockerfile] |
| PyTorch 在 Linux 精确要求的 NCCL | `nvidia-nccl-cu13==2.29.7` | [wheel 元数据][torch-metadata] |
| Foretoken P2P 候选选择的 NCCL | `nvidia-nccl-cu13==2.29.7` | 与上述 PyTorch 要求一致 |

插件要求 Python `>=3.10,<3.14`，可选运行时 extra 固定 `vllm==0.30.0`。
发行包名为 `vllm-afd-plugin`，import 名为 `afd_plugin`，注册入口为
`vllm.general_plugins` 中的 `afd = afd_plugin:register_afd`。基础包未声明
运行依赖；`--no-deps` 保留既有环境，但不代替依赖检查。见[包定义][plugin-package]。

插件通过 `setuptools-scm` 生成版本，构建时保留 Git 元数据，不能假定缺少
元数据的压缩包也可同样构建。CUDA 构建设置 `AFD_BUILD_ASCEND_OPS=0` 关闭
可选 Ascend 扩展。见[包定义][plugin-package]和[构建入口][plugin-setup]。

## NCCL 选择依据

固定 vLLM Dockerfile 为 DeepEPv2 GIN 选择 NCCL 2.30.7，该后端要求至少
2.30.4。官方 CPython 3.12、Linux amd64、CUDA 13.0 的 PyTorch wheel
精确要求 `nvidia-nccl-cu13==2.29.7`，PyPI 元数据也相同。见
[Dockerfile][vllm-dockerfile]、[wheel 元数据][torch-metadata]和[PyPI][torch-pypi]。

AFD 镜像选择 NCCL 2.29.7 满足固定 PyTorch 依赖，并保留普通依赖检查。
同步 P2P 插件使用 vLLM 的 `PyNcclCommunicator`，不依赖 DeepEPv2 GIN。
该选择仅影响 AFD 镜像，不修改普通 CUDA、MetaX 或 Omni 环境。见
[wheel 元数据][torch-metadata]和[P2P connector][plugin-p2p]。
上游用 `UV_OVERRIDE=/etc/uv-overrides.txt` 持久化 NCCL 选择；派生镜像
同步该文件，防止后续包安装重新选回其他版本。见[Dockerfile][vllm-dockerfile]。

## 原生运行边界

固定 README 的 1A1F 命令选择 `VLLM_USE_V2_MODEL_RUNNER=0`、DP=TP=1、
`--enable-expert-parallel`、`--enforce-eager` 和 `P2pNcclAFDConnector`。
两侧使用相同 rendezvous 和 rank 数，仅向 Attention 发请求。
`async_dp` 和 `compute_gate_on_attention` 默认均为 false。见
[README][plugin-readme]和[配置定义][plugin-config]。

connector 使用 TCP 集合 rendezvous，FFN ranks 排在 Attention 之前。
各方必须对端点、rank 数、模型形状、dtype 和角色分配达成一致。同节点
放置不证明跨 Pod GPU 通信成立，清理源码也不证明真实进程和 GPU 资源
正确释放。见[P2P connector][plugin-p2p]。

## 待补证据

官方发行记录列出 CUDA 13.0 的 `vllm/vllm-openai:v0.30.0`，CUDA 12.9
使用单独的 `v0.30.0-cu129`。tag 不是不可变产物。见[发行记录][vllm-release]。

- 不可变 `linux/amd64` 镜像 digest，以及实际 Python、vLLM、PyTorch、
  CUDA、NCCL 版本；本文未确定 OCI digest。
- [DeepSeek-V2-Lite][model-repository] 模型和 tokenizer 共用的固定
  revision；本文未确定该 revision。
- Foretoken 镜像实际构建成功，且实际 `FORETOKEN_VLLM_PYTHON` 环境依赖检查通过。
- 负责人执行 A100 1A1F 生成、有序退出、同节点跨 Pod 通信，保存请求输出、
  退出状态、遗留 GPU 进程和 Pod 状态。import 或端口连通不能替代验收。

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
