<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# 推理参数

[English](inference-parameters.md) | 简体中文

通过 `ModelService.spec.engineArgs` 配置所选引擎，参数名沿用原生名称，不写 `--`：

```yaml
spec:
  model: Qwen/Qwen2.5-0.5B-Instruct-AWQ
  backend: vllm
  engineArgs:
    quantization: awq
    dtype: half
    max-model-len: 8192
    tensor-parallel-size: 1
    gpu-memory-utilization: 0.85
```

值直接使用 YAML 布尔值、数字、字符串、列表或对象。未填写的选项沿用引擎默认值，`null` 表示不传该原生选项。具体取值需与引擎镜像、模型及硬件匹配。

## 选择模型服务节点

通过 Kubernetes 节点 label，将整个服务或某个 Pool 固定到指定机器组。为组内每台节点设置相同的 key/value：

```bash
kubectl label node <node-a> workload-group=group-a --overwrite
kubectl label node <node-b> workload-group=group-a --overwrite
kubectl get nodes -L workload-group
```

然后在模型 YAML 中使用同一组 key/value：

```yaml
spec:
  nodeSelector:
    workload-group: group-a
```

`ModelService.spec.nodeSelector` 选择默认 Pool，`modelPools[].nodeSelector` 为单个 Pool 选择另一组节点。Kubernetes 会把模型服务和运行时准备 Pod 调度到带有该 label 的节点。

## vLLM 常用参数

| 引擎参数 | 用途 |
| --- | --- |
| `max-model-len` | 输入和输出合计的最大 token 数 |
| `dtype` | 模型计算精度 |
| `quantization` | 权重量化方法 |
| `kv-cache-dtype` | KV Cache 精度，与权重量化分别配置 |
| `gpu-memory-utilization` | 每个引擎实例可使用的显存比例 |
| `max-num-seqs` | 每轮调度的最大序列数 |
| `max-num-batched-tokens` | 每轮调度的最大 token 数 |
| `enforce-eager` | `true` 时禁用计算图捕获 |
| `speculative-config` | 推测解码的原生配置字典 |
| `tensor-parallel-size` | 张量并行度（TP） |
| `pipeline-parallel-size` | 流水线并行度（PP） |
| `data-parallel-size` | 单个模型副本内的数据并行度（DP） |
| `prefill-context-parallel-size` | Prefill 上下文并行度（PCP） |
| `decode-context-parallel-size` | Decode 上下文并行度（DCP），复用已有 rank |

`nodes` 指定每个模型副本使用的 Kubernetes 节点数，`resources.requests.gpu.count` 是每个成员 Pod 申请的 GPU 数量。两者乘积必须等于 vLLM 的 TP × PP × DP × PCP，DCP 不增加 GPU 数。专家并行使用原生 `enable-expert-parallel`、`all2all-backend` 和 `enable-eplb` 参数。

模型副本可以跨节点运行，每个节点放置一个成员，按完整执行组启动、判断就绪和重启。`foretoken install` 自动准备 LeaderWorkerSet 控制器与 RDMA 分配，通信库从已分配设备中选择链路。跨节点使用的持久缓存需要所有成员均可访问。

P/D 和 E/P/D 各 Pool 可以在模型和引擎支持的组合内分别配置并行参数。PCP、DCP 的支持还取决于 attention backend。[EPD runtime 镜像](../examples/encoder-prefill-decode/README_zh.md) 包含支持上下文并行的 Mooncake 传输实现：Prefill 与 Decode 需要使用匹配的 PCP/DCP 缓存布局，TP 大小需互为整数倍。

填写 `modelPools[].engineArgs` 时，它会整体替换该 Pool 继承的服务级原生参数。服务副本数与引擎内部的数据并行度分别配置。

开启 EP 后，attention 可以按 TP × DP 执行，路由专家则分布在对应的 EP 组中。共享专家不会让 attention 的 KV Cache 在 DP ranks 之间共享。

### Attention/FFN Pool 配置

从 [DeepSeek-V2-Lite 候选示例](../examples/attention-ffn/README_zh.md)开始，
按安装、不可变模型准备、1A1F/2A2F 布局、请求到删除的路径操作。
其中的[验收状态](development/afd-acceptance_zh.md)不是已验证支持矩阵。

Aggregate vLLM Pool 的 `modelPools[].afd` 包含 `attention` 和 `ffn` 两侧。每侧分别填写 `nodes`、每成员 `resources` 和 `parallelism.dp`／`parallelism.tp`；节点数 × 每成员 GPU 数须等于 DP × TP。Pool 的 `replicas` 表示完整 Attention/FFN Group 的数量。填写 `afd` 后，不再填写 Pool 顶层的 `nodes` 或 `resources`，也不能用引擎参数覆盖角色拓扑或 `additional-config.afd`。

CUDA AFD profile 支持多成员及每成员多 GPU：每成员 GPU 数必须整除 TP 或被 TP 整除，Attention 的 DP × TP 不小于 FFN。同角色跨节点分散，跨角色同机只是偏好。完整成员全部就绪后，仅 Attention 成员零提供生成。多成员持久缓存须为 ReadWriteMany 且在所有选中节点可达。AFD 需要匹配的平台镜像/profile；依赖 RDMA 的扩展和自动整组恢复仍未闭环。参见 [AFD 工作负载边界](development/afd-group-workload_zh.md)，真实 GPU/Kubernetes 验收仍未完成。

## 推测解码

将完整的原生字典写在一起：

```yaml
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  engineArgs:
    speculative-config:
      method: ngram
      num_speculative_tokens: 2
      prompt_lookup_max: 4
```

方法和子字段沿用 vLLM。需要草稿权重时，`model` 可填写 Hub 模型 ID 或容器内可见的绝对目录，由 vLLM 下载、加载和缓存。`spec.source: modelscope` 同时适用于主模型和草稿模型的 Hub ID。

## 平台管理的选项

模型标识、启动端点、传输连接器和性能剖析由 Foretoken 管理，其余原生选项交给所选引擎解释。完整选项见 [vLLM 参数文档](https://docs.vllm.ai/en/latest/configuration/engine_args/)；当前已实现的 backend 为 vLLM。
