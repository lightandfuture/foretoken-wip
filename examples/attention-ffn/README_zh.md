<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# DeepSeek-V2-Lite Attention/FFN 分离部署

[English](README.md) | 简体中文

准备一个 Attention 和 FFN 分别运行在独立 GPU 进程中的 DeepSeek-V2-Lite 服务。
先从同节点各一个 Attention、FFN 成员（1A1F）开始，再考虑两节点 2A2F。
这些是**候选示例，不是通过实际部署的配置或已验证支持矩阵**。
按负责人要求，运行验证延期至服务器，未执行。
质量和生命周期等待补证据见[验收准备](../../docs/development/afd-acceptance_zh.md)。

## 安装并准备模型

以下命令供后续验证主机从仓库根目录执行。首版目标是 CUDA、两台 K3s 节点，
每台两张 NVIDIA A100 80 GB。先选一台有两张空闲 GPU 的节点；除下文模型资源外，
还要为准备任务、前端和平台预留 CPU、内存与存储。

1. 按[标准安装](../../cli/README_zh.md#安装-kubernetes-平台)使用**已发布且配套的 AFD 版本**。
   记录 CLI、Chart/CRD、controller、AFD 环境镜像和应用归档的版本与指纹，
   沿用安装入口：

   ```bash
   foretoken install
   ```

   本示例尚未提供真实 AFD 发布 digest。Chart 中自动生成的镜像名只是命名约定，
   不能证明镜像已发布。没有配套发布时，按
   [CUDA AFD 源码部署](../../docs/custom-deployment_zh.md#cuda-afd-源码部署)和
   [K3s/远程集群指南](../../docs/kubernetes-deployment_zh.md)准备固定源码。
   先用 `export` 将 `VLLM_AFD_BASE_IMAGE` 设为负责人记录的 CUDA 镜像 digest，
   将 `REGISTRY` 设为集群可达的仓库，再使用已有源码命令：

   ```bash
   pip install -e .
   foretoken install -e . --registry "$REGISTRY"
   ```

   需要覆盖配置时始终保留同一份 `--values`。普通 engine-source 覆盖层不会修改
   AFD 环境。启动配套 controller 前，先安装含 `status.afdMembers.memberIndex`
   及 role/memberIndex 联合列表键的 CRD；此顺序由平台安装入口负责。

   固定源码已收到一项同名冲突报告：显式覆盖恰好为
   `runtime.vllm.afd.image: afd-engine` 时，源码构建可能把该 OCI 名当成内部
   构建 stage，跳过镜像处理和来源记录。仅此路径暂停，等待所属负责人决定，
   不能泛化为全部镜像覆盖；本次任何源码构建路径均无运行验证。

2. 准备[共享模型存储](../../docs/model-storage_zh.md)。存储示例通过默认
   StorageClass 请求 100 GiB RWX PVC；默认类不能提供 RWX 时，在
   `storage/cache.yaml` 中选择真实可用的类。100 GiB 是起始请求，不是实测容量要求。

   ```bash
   kubectl apply -k examples/attention-ffn/storage
   kubectl -n foretoken-afd-demo get runtimecache models -o yaml
   ```

   由存储负责人向 `status.claimName` 指定的卷填入
   `deepseek-ai/DeepSeek-V2-Lite` 的完整**不可变快照**，位置是相对卷根的
   `models/deepseek-ai/DeepSeek-V2-Lite`。包含权重、配置、tokenizer 和必要模型代码。
   部署前记录真实模型/tokenizer commit 和文件指纹；这些是负责人待提供的输入，
   仓库没有编造 revision。准备任务、前端和全部成员必须看到同一份文件。
   创建 PVC 本身不会下载或上传这些本地文件。

   示例使用已有 `source: local` 契约。公开 ModelService 没有模型/tokenizer
   revision 字段，原生 `revision` 参数归平台管理。远程 `source: hf` 在各 Pool
   的准备记录中解析可变来源版本，并不因此证明不同 Pool 使用相同权重。
   参见[模型来源](../../docs/model-sources_zh.md)，服务期间不要替换本地快照目录。

3. 将 `NODE_A` 设为获准使用的第一台节点真实名称，确保**仅该节点**匹配
   `workload-group=afd-single`；示例没有私有主机名。

   ```bash
   kubectl label node "$NODE_A" workload-group=afd-single --overwrite
   kubectl get nodes -l workload-group=afd-single
   ```

## 部署、请求和观察

上述输入准备完成后：

```bash
RECIPE=examples/attention-ffn/same-node
foretoken deploy "$RECIPE" --timeout 30m
foretoken status "$RECIPE"
ENDPOINT="$(foretoken endpoint "$RECIPE" --timeout 10m)"
REQUEST_HOST="$(foretoken endpoint "$RECIPE" --host --timeout 10m)"
curl --fail "${ENDPOINT%/}/v1/completions" \
  -H "Host: $REQUEST_HOST" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite","prompt":"The capital of France is","max_tokens":32,"temperature":0,"seed":0,"return_token_ids":true}'
```

基础模型示例使用 Completions 和纯文本提示，不应用指令/聊天模板。保留真实 HTTP
响应和生成 token ID。`foretoken deploy` 与 `status` 观察的是控制面就绪，
不能证明请求成功或质量达标；此处没有把示意响应当成验收结果。
Gateway 用户按已有[网关指南](../../cli/README_zh.md#网关模式)配置前端 hostname。

```bash
foretoken status "$RECIPE" --watch
kubectl -n foretoken-afd-demo get modelservices,modelpools,modelgroups
kubectl -n foretoken-afd-demo get pods -o wide
kubectl -n foretoken-afd-demo get modelgroups -o yaml
```

核对全部预期成员的身份和 Ready 状态，不能只数 Running Pod。完整 Group 就绪后，
只有 Attention 成员零提供生成入口；其他 Attention 和 FFN 不是公开生成端点。
发现对象可以先于服务就绪存在。

## 选择布局

所有路径复用同一 namespace 和服务名称，**每次只选一个**，不能当作并行部署。
每个 A/F 成员请求一张 GPU、4 CPU、48 GiB 内存。BF16 和 2,048-token 上下文是
显式示例设置，不是性能推荐值。

| 路径 | A 成员 × 每成员 GPU；DP/TP | F 成员 × 每成员 GPU；DP/TP | 完整 Group | 总 GPU | 放置前提 |
|---|---|---|---:|---:|---|
| `same-node` | 1 × 1；1/1 | 1 × 1；1/1 | 1 | 2 | 仅一台符合 selector 的节点，有两张空闲 GPU |
| `two-node` | 2 × 1；2/1 | 2 × 1；2/1 | 1 | 4 | 恰好两台符合 selector 的节点，各两张空闲 GPU |
| `two-groups` | 每组 1 × 1；1/1 | 每组 1 × 1；1/1 | 2 | 4 | 两台节点；跨节点布局依赖受控容量 |
| `baseline` | 普通 vLLM，单成员，默认 DP=TP=1 | — | 1 | 1 | 一台符合 selector 的节点；相同不可变模型和 BF16 |

选择 `two-node` 时，将 `NODE_A`、`NODE_B` 设为两台获准节点，
都标记为 `workload-group=afd-pair`，确认没有其他节点匹配。
此前部署实际释放资源后再选择：

```bash
kubectl label node "$NODE_A" "$NODE_B" workload-group=afd-pair --overwrite
kubectl get nodes -l workload-group=afd-pair
RECIPE=examples/attention-ffn/two-node
foretoken deploy "$RECIPE" --timeout 30m
```

同一 Group 的同角色成员按 hostname 硬分散。只有两台符合条件的节点、每台两张 GPU
时，2A2F 在每台放一个 A 和一个 F。跨角色亲和只是软偏好，既不普遍保证同机，
也不保证 A0/F0 或 A1/F1 同序号配对。Pool 的 `replicas` 表示完整 AF 实例，
不是角色 DP 或成员数。

`two-groups` 用于两个独立 1A1F Group 的隔离检查。静态 `replicas: 2` 无法保证
两组都跨节点。该检查需要负责人先让每台恰好一张 GPU 空闲，启动第一组，再仅释放
事先登记的占位容量，让第二组再次看到每台一张空闲 GPU。扩为两组时保留第一组 UID，
记录实际放置。不得挤占无关工作负载，也不能虚构 role selector 或序号配对规则。
受控步骤仍待[生命周期验收](../../docs/development/afd-acceptance_zh.md)。

## 更新和删除

计划更改配置/代码时，保留相同不可变输入，编辑选中的配置，再沿用显式重新部署：

```bash
foretoken deploy "$RECIPE" --timeout 30m
foretoken status "$RECIPE"
```

更新前核对可用容量和旧 cohort 的退休过程。四张 GPU 全部占用时，2A2F 替换依赖
旧 GPU 释放后新成员才能运行；这条生命周期尚未验证。额外容量不是已确立的前提，
也不能替代安全退休。
按可能中断安排操作，不承诺无中断更新、透明恢复或安全旧镜像回退。把这些命令用作
生产流程前，需完成负责人控制的更新/删除检查。

```bash
foretoken delete "$RECIPE" --timeout 30m
```

删除目标是前端和模型服务。存储与 namespace 单独准备，因此保留到数据负责人明确
退役。对象消失、CLI 成功退出、finalizer 移除或单独的 `terminationConfirmed`
都不能证明进程停止、GPU 释放。另一布局复用容量前保留真实进程/GPU 证据，不能
强删 finalizer。阅读既有[升级/回退边界](../../docs/development/source-image-lifecycle_zh.md#afd-环境与应用选择)。

## 限制和后续测量

- 多成员持久存储必须 RWX 且全部选中节点可达。非 RWX 卷的 required 同机亲和与
  同角色分散冲突时会拒绝创建，不能删除缓存约束来绕过。
- 非空 RDMA profile 配合多成员或每成员多 GPU 时，由于 AFD 资源投影未完成而
  明确拒绝。安装自动发现 RDMA 不会补齐投影；不能清空 RDMA 配置让示例通过，
  应报告依赖。即使 1A1F 也不能据此认定 RDMA 已正确分配。
- 调度 Pending、准备失败和生成失败是不同状态。修改输入前先检查 status/events、
  节点容量、PVC 挂载和镜像/应用身份。实际集群仍需证明发现与 NCCL 的节点间
  网络可达，不承诺传输回退。
- 可信停止、foreground GC 顺序、节点失联分类、根因保留、自动退休/替换以及
  部分多 Pool 更新仍有缺口；触发条件及暂停结论见
  [验收准备](../../docs/development/afd-acceptance_zh.md)。

复用现有[greedy/分布比较](../../benchmarks/docs/eval/distribution-comparison_zh.md)、
[任务质量](../../benchmarks/docs/eval/README_zh.md)和
[性能](../../benchmarks/docs/perf/README_zh.md)入口。
[AFD 测量准备](../../docs/development/afd-acceptance_zh.md#质量和性能)
记录真实输入/输出契约以及请求 seed 缺口。质量与 TTFT/TPOT 分开记录，不设首版 SLO/SLA。
