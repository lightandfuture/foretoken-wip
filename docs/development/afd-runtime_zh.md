<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# 构建与验证 CUDA AFD 镜像

[English](afd-runtime.md) | 简体中文

构建显式选择的 CUDA 环境，用于原生 Attention/FFN 分离（AFD）验证。
镜像提供插件，不包含 Foretoken AFD 启动协议、调度或就绪实现。
候选环境必须由负责人完成 A100 和 K3s 验收，才能认定为受支持环境。

## 在 Linux amd64 上构建

在仓库根目录使用支持 BuildKit 的 Docker 和 Make。先在构建服务器解析
官方 CUDA 13.0 发行镜像的 digest，并保留该引用：

```bash
docker pull --platform linux/amd64 vllm/vllm-openai:v0.30.0
AFD_BASE_IMAGE=$(docker image inspect vllm/vllm-openai:v0.30.0 \
  --format '{{index .RepoDigests 0}}')
AFD_IMAGE=foretoken-model-server-afd:$(git rev-parse --short=12 HEAD)
make image-model-server-afd \
  VLLM_AFD_BASE_IMAGE="$AFD_BASE_IMAGE" \
  AFD_MODEL_SERVER_IMAGE="$AFD_IMAGE"
```

该目标先构建 `foretoken-vllm-afd:dev`，再复用现有 model-server 构建入口
包装最终镜像，两步均使用 `linux/amd64`。可用 `VLLM_AFD_IMAGE` 指定其他
中间镜像名；`IMAGE_TARGET=environment` 和 `IMAGE_OUTPUT` 仍用于最终
构建。普通 model-server、MetaX、Omni 构建不选择 AFD，平台默认值不变。

AFD 目标默认使用 `/usr/bin/python3`；显式设置 `FORETOKEN_VLLM_PYTHON`
可为两步构建选择其他解释器。插件 wheel 独立构建后安装到该环境。
构建核对 vLLM 0.30.0、PyTorch 2.13.0/cu130、
CUDA 13.0.3 和 NCCL 2.29.7，执行 `uv pip check`，并将测得版本写入
`/usr/share/foretoken/afd-runtime.json`。NCCL 选择同时更新继承的 uv
override，供后续包安装使用。该同步 P2P 候选环境不支持 DeepEPv2 GIN。
版本依据见[固定来源](afd-runtime-sources_zh.md)。

Rust workspace 保留独立的 vLLM gitlink。`VLLM_REVISION` 控制 Rust
构建源码，不是 Python 引擎版本；不要为此镜像升级该 gitlink。

## 核对构建产物

```bash
docker image inspect "$AFD_IMAGE" \
  --format 'id={{.Id}} platform={{.Os}}/{{.Architecture}} labels={{json .Config.Labels}}'
docker run --rm --entrypoint cat "$AFD_IMAGE" \
  /usr/share/foretoken/afd-runtime.json
docker run --rm --entrypoint sh "$AFD_IMAGE" -c \
  '"${FORETOKEN_VLLM_PYTHON:-python}" -m pip check'
```

预期为固定插件 revision、上述版本、目标 Python 路径，且没有依赖错误。
这些是构建检查，不是推理验收。最终 OCI manifest digest 与本地 image ID
分别记录，后者是配置 digest；同时保留源码 SHA、输入镜像引用、构建命令
及日志。

## 负责人：A100 原生 1A1F

准备两张 A100 80GB、兼容 CUDA 13.0 的 NVIDIA 驱动、NVIDIA Container
Toolkit 和已构建镜像。从官方 `deepseek-ai/DeepSeek-V2-Lite` 仓库选定
一个完整 commit，模型和 tokenizer 均使用它并记录。镜像本身不确定模型
revision。将 `MODEL_CACHE` 设为服务器现有缓存目录，在每个验证容器中
传入缓存与 `MODEL_REVISION`。在两个宿主机终端导出：

```bash
export MODEL_REVISION='<full DeepSeek-V2-Lite commit>'
export MODEL_CACHE='<existing absolute model-cache directory>'
```

用同一镜像启动两个容器，各分配一张不同 GPU，使用
`--network host --ipc host` 并挂载模型缓存。将镜像 entrypoint 改为 `sh`。
在容器内运行下面命令，FFN 使用 `ROLE=ffn`、`HTTP_PORT=18001`；Attention
使用 `ROLE=attention`、`HTTP_PORT=18000`。两侧都启动后才能完成 rendezvous。
在第一个终端执行：

```bash
docker run -it --name foretoken-afd-ffn --gpus device=1 \
  --network host --ipc host -v "$MODEL_CACHE:/root/.cache/huggingface" \
  -e MODEL_REVISION -e ROLE=ffn -e HTTP_PORT=18001 \
  --entrypoint sh "$AFD_IMAGE"
```

在第二个终端执行：

```bash
docker run -it --name foretoken-afd-attention --gpus device=0 \
  --network host --ipc host -v "$MODEL_CACHE:/root/.cache/huggingface" \
  -e MODEL_REVISION -e ROLE=attention -e HTTP_PORT=18000 \
  --entrypoint sh "$AFD_IMAGE"
```

分别在两个容器内执行：

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

保持 PP=PCP=DCP=1。初始化完成后向 Attention 发送请求：

```bash
curl --fail-with-body http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v2-lite-afd","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

预期生成非空文本，无集合通信错误或请求挂起。import 成功或端口打开不算
通过。请求结束后先停止 Attention，再停止 FFN，保留日志与实际退出码。
确认没有强制 kill，且 `nvidia-smi` 不再显示遗留引擎进程或 GPU 占用，
最后删除验证容器。在第三个宿主机终端执行：

```bash
docker stop --time 60 foretoken-afd-attention
docker stop --time 60 foretoken-afd-ffn
docker inspect foretoken-afd-attention foretoken-afd-ffn --format '{{json .State}}'
docker logs foretoken-afd-attention
docker logs foretoken-afd-ffn
nvidia-smi
docker rm foretoken-afd-attention foretoken-afd-ffn
```

## 负责人：同节点跨 Pod 通信

在同一 A100 节点创建两个临时 Pod，各申请一张 GPU，以相同镜像 digest 和
固定模型 revision 运行原生命令。记录 K3s server 版本、Pod 清单、所分配
GPU、网卡及 NCCL 环境。两侧共享的 rendezvous 地址必须指向可达的 FFN Pod；
仅共享网络命名空间时才能使用 `127.0.0.1`。Attention 的 HTTP 监听地址
需允许验证网络访问。

并行启动两侧，重复生成和有序退出。预期跨 Pod 真实交换张量并释放资源。
保存请求输出、通信日志、容器及 Pod 退出状态、终止后的 GPU 占用。
host-network 成功不能替代跨 Pod 验证。这是原生验证；Foretoken 管理的
AFD 生命周期和控制器部署不属于镜像交付范围。
