<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# AFD 候选范围与验收准备

[English](afd-acceptance.md) | 简体中文 · [部署示例](../../examples/attention-ffn/README_zh.md)

本文为负责人执行 DeepSeek-V2-Lite AFD 验收准备输入，**不是已验证支持矩阵或首版
验收完成记录**。截至 2026-10-12，按负责人要求，运行验证延期至服务器，未执行。
本地源码自审不等于独立审阅或运行证据；最终审阅仍须覆盖集成后的结果树。

## 候选范围与证据

准备依据是 commit `e0a9f8331110a65bd03f7a50f3c211e18a5a0ad1`，
tree `cec217a80b3ba76f4850260fec50f8afbbcb1ec9`。
本地示例/文档提交没有补写运行时或控制器修复。

| 领域 | 计划 / 源码实现 | 证据状态 |
|---|---|---|
| 首版硬件/模型 | CUDA、DeepSeek-V2-Lite、两台 K3s 节点各两张 A100 80 GB；1A1F 起步并交付 2A2F | 仅为目标；真实服务器版本、模型/tokenizer commit、部署证据缺失 |
| 运行时 | [固定插件与 Python vLLM](afd-runtime-sources_zh.md)、Go V2 投影、Rust 唯一 argv；保留普通 V1 | 候选实现；前置源码审阅不证明 GPU 或旧运行时兼容 |
| Group 调度和服务 | [UID/role/memberIndex 身份、完整 Ready 集合、A0 生成、A0/F0 coordination 与共享 F0](afd-group-workload_zh.md) | PR-09 已独立源码审阅；三种布局未运行 |
| 安装/发布 | 已有安装器交付[独立 AFD 环境和应用](source-image-lifecycle_zh.md#afd-环境与应用选择) | PR-10 独立源码审阅已归档：PR10-F1/P2 局部阻断同名显式镜像路径；E1–E7 证据缺口保留；没有发布 digest 或安装证据 |
| 示例 | 共享本地模型输入；1A1F、2A2F、两个完整 1A1F Group 和普通基线 | 本地准备；未渲染 Kustomize，未实际部署 |
| 质量/性能 | 下述已有 eval/perf 消费路径 | 未执行；seed 和 AFD 环境快照缺口保留 |
| 恢复/更新/删除/回退 | 原生命周期 owner 及下述依赖 | 安全结论暂停；不承诺透明恢复或无中断 |

模型注册和拓扑算术被接受，不等于部署支持。后续每个用例单独标记未执行、通过、失败
或依赖暂停，并附来源与证据；不能把缺失结果填成成功。

## 责任与固定输入

CLI/Chart 选择配套环境和应用；ModelService 拥有公开意图；compiler/resolver 产生
Pool/Group 契约；preparation 负责模型获取和来源记录；Group reconciliation 负责
成员、发现、服务发布与清理；Rust 渲染私有计划并执行引擎；benchmark/eval 消费已有
服务和输出接口，不成为第二个部署控制器或评估器。

服务器工作前，由负责人记录：

- 最终源码 commit/tree，配套 CLI/Chart/CRD/controller 版本、AFD 环境 OCI digest、
  应用归档指纹、普通基线引擎 digest，以及实测 Python/vLLM/plugin/CUDA/NCCL 版本。
  遵循已有[发布 owner](release_zh.md#cuda-afd-发布输入)，命名不等于发布。
- 真实 K3s/Kubernetes、driver、GPU/节点清单，selector 匹配节点、空闲容量、网络
  可达性和 RDMA profile。记录拒绝组合时保留原 profile，不能禁用以绕过限制。
- 相同 DeepSeek-V2-Lite 模型/tokenizer revision、config/模型代码指纹、BF16
  （或两侧一致的 FP16）、存储来源和挂载。公开 API 不提供显式 provider revision，
  因此示例使用不可变本地文件；规范化 revision `local` 不是文件指纹，需单独记录。
  控制器拥有的远程 `source.json` 固定一个 Pool revision，不能证明独立基线/候选一致。
- 输入语料/任务版本与指纹、实际 prompt token 或聊天模板、解码参数、seed、客户端
  依赖和负载设置。两侧引擎与硬件版本可以不同，但必须分别记录。

## 服务器顺序

以下全部是**操作准备，不是本次交付执行过的命令**。

1. 将最终固定代码放到验证主机，运行本地延期的仓库、Go/Rust、Helm 和 Kustomize
   检查，复用其所属命令，不新建验证驱动或永久 CI 门禁。确认含 role/memberIndex
   联合列表键的 CRD 先于配套 controller 安装。
2. 按[示例最短路径](../../examples/attention-ffn/README_zh.md)：标准发布或已登记
   源码安装、创建并填充不可变共享存储、选择单节点、部署 1A1F、请求、观察状态。
   保留渲染后的公开输入、实际镜像/应用身份、完整 Group/member/Pod UID、节点、
   HTTP 响应和 token ID。
3. 2A2F 使用恰好两台符合 selector 的节点，各两张空闲 GPU。记录两个 A、两个 F，
   每角色 DP=2/TP=1、每成员一张 GPU、共四张。核对同角色分散、全部 runtime Ready
   身份和仅 A0 生成，角色 coordination 与共享 F0 分离且限定同一 UID，不能由同机
   推断 A/F 序号配对。
4. 两个独立 1A1F 共用两节点时，在每组调度前控制容量，使每台恰好一张 GPU 空闲。
   只使用负责人批准且事先登记的占位资源，第一组放置完成后才释放其占位容量，
   再增加第二组。只改 desired replicas 并保留第一组 UID；显式 deploy 还会重新选择
   当前应用，不能默认当作纯扩副本。记录 UID/role/index/Pod 身份、发现 selector、
   实际布局和独立请求。静态 `replicas: 2` 不保证这种跨节点布局。该受控检查仍待
   容量和生命周期决策，不提供占位脚本。
5. 解决下述阻塞后，使用已有入口分别测量质量与性能。失败样本必须记为失败。
   差异分析及接受依据由负责人提供，性能改善不能替代质量验收。
6. 负责人控制的生命周期阶段，覆盖更新、正常删除、启动失败、成员 fatal、节点
   失联和 foreground 删除，记录 admission/撤销、在途完成或报错、真实进程组/
   子进程及 GPU 分配，保留前后 UID 和容量。对象删除、超时、信号或终止标记都
   不足以证明停止；所属缺陷与证据未闭合前，不承诺恢复或回退安全。

## 质量和性能

复用已有[比较指南](../../benchmarks/docs/eval/distribution-comparison_zh.md)和
[任务指南](../../benchmarks/docs/eval/README_zh.md)，不增加第二评估器。真实消费契约：

| 阶段 | 生产者 → 消费者 | 实际行为 / 输出 |
|---|---|---|
| 分发 | `evaluation.py` → `config/evaluation.py` → `config/distribution_comparison.py` | `--reference`/`--reference-url` 选择比较；`--greedy-compare` 选择生成序列；比较入口没有 `--seed` |
| 输入 | `runs/distribution_comparison.py` | 顺序取原始文本/JSONL text 窗口，tokenizer 有 BOS 时加入；两侧使用相同 token-ID 前缀，不应用聊天模板 |
| 请求 | `integrations/distributions.py` 的 `CompletionComparisonClient.generate` | Completions 发送 token-ID prompt、max_tokens、temperature=0、stream=false、return_token_ids=true；**不发送 seed** |
| 响应 | Frontend Completions 输出 → 比较客户端 | 需要原生 `choices[0].token_ids`、text、正常 stop/length finish_reason；不重新 tokenize 生成文本 |
| 序列结果 | Run → `results/distribution_comparison.py` | `greedy_comparison_candidates.csv` 含 valid_samples、failed_samples、exact_matches、exact_match_rate；`greedy_comparison_samples.csv` 含 token IDs、结束原因、mismatch_count、从零计数的 first_divergence（长度不同取较短长度）；失败不进入有效样本一致率 |
| 其他输出 | 已有质量/分布 publisher | `metrics.json` 含 prompt_windows/protocol；`serving_environments.json` 记录可得环境；任务输出 metrics.json/native/evaluator.log；分布输出候选 CSV 和逐位置 JSONL |

默认分布比较是 full-vocabulary logprobs 的 teacher forcing，不是 greedy continuation
或任务质量；兼容引擎需设置 `max-logprobs: -1`，最小示例没有启用。Greedy 比较
不支持 `--resume`。temperature=0 不能补足同 seed 证据；两侧原文输入是有意采用
的共同输入，不应被误写为输入不同。

两个具体依赖限制当前操作准备：

1. **Seed：**比较配置没有 seed 字段，请求生产者不发送 seed，protocol 输出不能
   证明请求使用同一 seed。前端接受 seed 并不能修复消费者。严格同 seed 验收仍待
   benchmark owner 决策/修复。
2. **AFD 环境快照：**`results/environment.py:serving_environment` 直接索引
   `group.spec.resources` 与 `parallelism`；AFD Group 生产者省略两项，资源/
   拓扑在 `spec.afd`。Kustomize AFD 目标在评估前可触发未捕获 KeyError，错误边界
   只捕获 DeploymentError。受影响 eval/perf 路径依赖暂停，本 docs 分支不修复。

已有 `--url` 来源不操作 Kubernetes，也不进入该 Group 快照路径；仅作为**尚未执行
的准备候选**，需要人工补齐环境证据和两个独立识别的真实服务端点，仍不解决 seed
缺口。先将 `BASELINE_URL`、`AFD_URL` 设为真实 Completions URL，
`TOKENIZER_DIR` 设为客户端固定 tokenizer/config 目录，`CORPUS` 设为负责人选择的
文本/JSONL 语料，再使用示例命令：

```bash
foretoken eval --url "$AFD_URL" --model deepseek-ai/DeepSeek-V2-Lite \
  --reference-url "$BASELINE_URL" --reference-model deepseek-ai/DeepSeek-V2-Lite \
  --tokenizer-path "$TOKENIZER_DIR" --dataset "$CORPUS" \
  --greedy-compare --context-length 128 --num-windows 4 --max-tokens 64 \
  --output local
```

这些样本数量是准备值，不是验收阈值。URL 模式不会自动填 Gateway Host header，
必须提供可正确路由的 hostname/HTTPS URL；环境输出也没有权威引擎/硬件清单，
需另外附源码/digest/拓扑。示例变体共用名称，直接把目录当成不同存活服务传入，
可能原样复用同一个现存部署；不能把 AFD 和自身比较。顺序切换依赖可信资源释放，
四张 GPU 不能同时容纳 2A2F 和普通基线。

任务评分由负责人通过已有原生 evaluator 选项，选择相同 evaluator/task 版本、
样本 ID、模板、解码和 seed 设置。记录完整解析配置及请求，再保存 `metrics.json`、
`native/`、`evaluator.log`。例如 `foretoken eval --url` 配合
`--evaluator lm-eval --tasks piqa --limit 100 --output local` 使用已有似然路径；
按[任务指南](../../benchmarks/docs/eval/README_zh.md#候选答案似然与困惑度)填写真实
模型和客户端 tokenizer。原生任务 RNG 设置本身不证明服务请求 seed 相同。

TTFT/TPOT 使用已有 streaming performance 入口，两侧保持相同 prompt/负载。
将 `SERVICE_URL` 设为对应真实服务的 Chat Completions URL，`PROMPT` 为负责人
批准的固定文本：

```bash
foretoken perf --url "$SERVICE_URL" --model deepseek-ai/DeepSeek-V2-Lite \
  --prompt "$PROMPT" --num-prompts 20 --max-concurrency 1 --max-tokens 64 \
  --temperature 0 --extra-body '{"seed":0}' --stream --output local
```

基础模型示例没有提供已验证聊天模板。使用这条基于 chat 的性能命令前，负责人必须
先提供并确认两侧相同模板；该前提仍待满足。这组负载与原始 token greedy 协议不同。
保留成功/错误、[TTFT/TPOT 定义](../../benchmarks/docs/metrics.md)、输入/输出长度、
负载和硬件。命令本身不形成 SLO/SLA 或性能改善结论。

## 暂停的生命周期结论

PR-10 独立审阅另报告源码构建镜像同名冲突：显式
`runtime.vllm.afd.image: afd-engine` 可能与 `auto` 产生的内部 stage 混淆，
跳过本地镜像处理及来源记录。完整独立报告已归档，唯一确认问题为 PR10-F1/P2，
第二轮没有额外独立 finding。完整安装/发布链的 E1–E7 证据缺口及最终集成树审阅
G11 仍未完成。仅此同名路径暂停，
不能据此认定其他覆盖也有该问题或已经验证。本次文档交付不包含前置返修。

| 依赖 | 触发条件与影响 | 待补证据 |
|---|---|---|
| EX-003/F1 | PodGC 合成终态、container state 为空/Waiting 时，可能写 terminationConfirmed 并放行 finalizer，缺可信物理停止 | 停止证据 owner 修复；进程/子进程/GPU 观察 |
| F2 | Group controller dependent 的 foreground GC 可绕过预期 A → F → discovery 顺序 | 清理 owner 修复及真实 foreground 删除证据 |
| F3 | 节点失联可能被分类为 MemberDeleting，丢失节点不可用语义 | 节点失联分类及不误认终止 |
| EX-004 | 非 Failed 观察根因可被 ServingReconcileFailed 覆盖 | 撤销中保留角色/成员根因 |
| Group 恢复 | 失败同 spec Group 可被保留，自动退休/完整替换未闭环；W1/W2 未执行 | 可信停止 → 退休 → 新 UID 替换与实测 |
| 多 Pool 更新 | 更改后缩为零的 Pool 可能保留旧 serving 容量，另一 Pool 等待这些 GPU | 跨 Pool 容量/提交推进及真实释放 |
| RDMA/cache | 非空 RDMA 与扩展拓扑拒绝；非 RWX required hostname affinity 与多成员分散冲突 | 所属投影修复及真实兼容存储/网络 |

Repair `1776fe081373e6d4694370100b01ec077ba29b89` 不在固定源码祖先链，
这里没有应用或宣称其 ownerReference 布局。源码审阅报告不能关闭这些依赖。
恢复、退出、更新和回退安全结论继续暂停；独立示例/文档准备继续推进。

## 文档参考

2026-10-12 读取：[KServe 0.20 LLM 部署](https://kserve.github.io/website/docs/getting-started/genai-first-isvc)
和 [Helm 4.3.0 快速入门](https://helm.sh/docs/intro/quickstart/)。
参考其读者顺序：准备与已有平台、最小服务、真实请求、状态、清理；替代安装方式链接
到所属指南。其 API、基础设施及验收结论不是 Foretoken 输入。
