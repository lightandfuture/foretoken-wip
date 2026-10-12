<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# Managed AFD 运行时契约

[English](afd-managed-runtime.md) | 简体中文

model-server 消费私有 V2 plan，支持同步 CUDA AFD 角色跨多成员或单成员多 GPU。
OCI 和 A100 验收仍由负责人执行。[Group 物化](afd-group-workload_zh.md) 负责多成员发现、角色放置和完整集合发布，整组恢复
仍未闭环。使用运行时源码构建[固定候选镜像](afd-runtime_zh.md)。

## 启动契约与生命周期

普通 V1 保留 nodeCount 及原有 LWS 环境语义。V2 禁止 nodeCount，由 afd 提供 role、
memberIndex、memberCount、groupUid、共享 F0 host/port、numAttentionRanks 和
numFfnRanks。两侧共享新 Group UID，rendezvous 只能定位承载 F0 的成员。
多成员角色还须提供含 host、masterPort、handshakePort 的 afd.coordination 对象。

Group 物化方提供按 UID 隔离、启动发现不依赖 Ready 的角色成员零端点。masterPort
协调原生 worker，handshakePort 连接该角色 EngineCore 与 Rust leader；共享
跨角色 afd.host/port 独立定位 F0。运行时只消费端点，不创建发现对象。物化方须预留
不冲突端口，允许原生 peer 流量及动态传输端口。共用网络命名空间的角色须使用不同
角色端口，端点不能负载均衡到其他成员或 Group。单成员省略 coordination，保留
原生本地协调。

BuildAFDLaunchPlan 是 Group 工作负载 owner 为每个角色/序号调用的 Go 投影边界。
compiler 接受下述拓扑，控制器检查存储条件并观测完整集合。运行时 V2 字段保持不变；
控制器 status 按角色/序号标识成员。

每角色满足 memberCount × 每成员 GPU 数 = DP × TP。每成员 GPU 数须整除 TP
或被 TP 整除，与固定原生同构放置一致。Attention ranks 不少于 FFN，不要求整数比例。
V2 保持 PP=PCP=DCP=1、EP 开启，无 KV/EC 传输、KV 事件、EPLB 或原生诊断。
Rust 唯一渲染同步 P2pNcclAFDConnector、async_dp=false、compute_gate_on_attention=false、
V1 model runner 和 eager 执行。保留其他合法 additional-config 内容，拒绝用户 afd
与原生拓扑覆盖，只发一个合并 JSON 参数。原生别名和缩写不能覆盖平台拓扑。

| 身份 | 生产者与消费者 |
|---|---|
| 角色 memberIndex/memberCount | 物化方 → V2 → node-rank/nnodes，不取物理节点去重数 |
| 每成员 GPU 数 G | 规范化资源；Rust 从 DP × TP / memberCount 推导 |
| DP 放置 | 固定 vLLM：本地 DP=max(G / TP, 1)，起始全局 DP rank=floor(memberIndex × G / TP) |
| TP rank | 原生 worker 进程组，可跨成员 |
| AFD role rank | 固定插件：全局 DP rank × TP + TP rank（PCP=1） |
| AFD world rank | FFN 为 role rank；Attention 为 numFfnRanks + role rank |
| Group UID | 已持久化 Group → plan、发现隔离和就绪键 |

Rust 核对精确本地集合 [memberIndex × G, (memberIndex + 1) × G)，Pod 序号不能代表
全部 worker。固定 vLLM 参数实现拥有 DP 放置，插件 topology 拥有 AFD rank 映射。
V2 身份来自 plan，多成员 Pod 的可路由地址由 FORETOKEN_MEMBER_IP 提供。
普通 V1 仍解释 LWS_WORKER_INDEX、LWS_LEADER_ADDRESS 和 Group UID 环境。
AFD 不进入普通 MetaX 网络校准 launcher。

适配器随既有 Python 源码包分发，使用 `vllm.general_plugins` 入口；没有私有
状态 socket 环境时不激活。运行时保留显式 `VLLM_PLUGINS` 列表，并加入两个
必需的 AFD 插件。

## 真实就绪与失败

固定插件为 `b7561d726b92f366f92634773b731b38e364fbe6`，Python vLLM 为
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`（0.30.0）。薄适配器委托插件
[P2P connector][p2p-source] 和 [FFN worker][ffn-source] 的真实方法执行。
固定 vLLM 的 [worker 包装层][worker-source] 在构造 worker 之前加载通用插件：

1. 每个 worker 完成实际 connector 初始化。
2. FFN rank 设置 CUDA device 后进入 recv_dp_metadata_list，在既有 AFD subgroup
   store 写 UID/rank 键并上报 loop_ready。创建线程本身不代表就绪，FFN 握手可早于循环。
3. 每个 Attention worker 等待其映射的 FFN subgroup 键，再上报 connector_ready；
   包含未加入较小 metadata group 的 worker。
4. Attention 在实际原生 compile/warmup 完成后上报 worker_ready。
5. Rust 要求全部预期本地 role rank 的 connector 与执行事件，每 rank 保留独立通道。
   Leader 还须完成全部 DP EngineCore 握手；原生 executor 初始化并监视含远端的 TP worker。

沿用既有状态响应：Group UID、角色、成员索引/数量和聚合成员 ready。启动完成后才
提供 HTTP 状态，连接失败或超时表示不可用。成员就绪不能代表完整 Group 集合。
仅 Attention 成员零提供生成和启动时关闭的 serving gate；其他 Attention 成员及
所有 FFN 仅提供生命周期/状态接口。Group owner 必须观测完整集合再打开 admission，
见[服务就绪](afd-group-workload_zh.md#服务就绪与入口撤销)。不引入第二状态或恢复协议。

prepare/resolve 保留共享模型/缓存准备生命周期，不启动引擎。
同一个 startup deadline 覆盖准备、引擎启动和 AFD 就绪。插件 world rendezvous
及就绪键等待使用同一 lifecycle timeout；Rust 外层截止仍控制插件内部固定超时
操作。FFN 致命错误、状态通道断开、EngineCore 不健康或子进程退出均终止本成员。
AFD 不走普通实例的临时缓存本地重试，恢复必须使用新完整 Group 和通信域。

Attention 先排空请求，再停止其受管进程组；先停 Attention，再停 FFN。
通用 Rust upstream 补丁在 launcher 被回收后仍保留 PGID，遵守调用方排空期限，
仅在 Tokio 回收 launcher 后回收同进程组内被 PID 1 收养的后代，
SIGKILL 后确认整个组结束。强制退出另有五秒终止确认期限，未结束则返回错误。

## 负责人验收

沿用镜像指南中的同一镜像 digest、固定模型/tokenizer revision、缓存和两张 GPU。
保存两侧实际 plan。Attention 最小输入与英文文档中的 JSON 完全相同：

```json
{
  "version": 2,
  "afd": {
    "role": "attention", "memberIndex": 0, "memberCount": 1,
    "groupUid": "acceptance-run-001", "host": "127.0.0.1", "port": 6239,
    "numAttentionRanks": 1, "numFfnRanks": 1
  },
  "artifacts": {
    "model": "deepseek-ai/DeepSeek-V2-Lite", "source": "hf",
    "revision": "<fixed model commit>",
    "tokenizer": "deepseek-ai/DeepSeek-V2-Lite",
    "tokenizerRevision": "<same fixed commit>"
  },
  "parallelism": {"tp": 1, "pp": 1, "dp": 1, "pcp": 1, "dcp": 1,
                  "ep": {"backend": "", "eplb": false}},
  "kv": {"kind": "none", "events": false},
  "lifecycle": {"startupSeconds": 600, "drainSeconds": 60},
  "internalGenerateRequestBodyLimitBytes": 1048576,
  "engineArgs": {"dtype": "bfloat16", "max-model-len": 2048,
                 "trust-remote-code": true}
}
```

将 JSON 保存为 `attention.json`，将两个 revision 占位符替换为已记录的模型
完整 commit，并为本轮使用新 `groupUid`。以下命令需要宿主机 Python 3，沿用镜像
指南中的 `AFD_IMAGE`、`MODEL_CACHE`；两容器共享宿主网络，因此可使用 loopback：

```bash
python3 - <<'PLAN'
import json
from pathlib import Path
plan = json.loads(Path("attention.json").read_text())
plan["afd"]["role"] = "ffn"
Path("ffn.json").write_text(json.dumps(plan))
PLAN

docker run -d --name foretoken-managed-afd-ffn --gpus device=1 \
  --network host --ipc host -v "$MODEL_CACHE:/root/.cache/huggingface" \
  -e "FORETOKEN_VLLM_LAUNCH_PLAN=$(cat ffn.json)" \
  -e FORETOKEN_INTERNAL_LISTEN=0.0.0.0:18001 "$AFD_IMAGE"
docker run -d --name foretoken-managed-afd-attention --gpus device=0 \
  --network host --ipc host -v "$MODEL_CACHE:/root/.cache/huggingface" \
  -e "FORETOKEN_VLLM_LAUNCH_PLAN=$(cat attention.json)" \
  -e FORETOKEN_INTERNAL_LISTEN=0.0.0.0:18000 "$AFD_IMAGE"
```

查看两侧 `docker logs` 等待完成启动；启动阶段连接失败属于预期。然后执行：

```bash
curl --fail http://127.0.0.1:18000/internal/afd/status
curl --fail http://127.0.0.1:18001/internal/afd/status
```

先核对两侧响应与保存的 plan 身份一致且均 ready，再由此独立验收对的负责人
显式打开 Attention 入口；Kubernetes 部署由 Group 控制器执行这一步：
若本次 plan 使用不同的 `groupUid`，以下请求也应使用该实际值。

```bash
curl --fail-with-body http://127.0.0.1:18000/internal/afd/serving \
  -H 'Content-Type: application/json' \
  -d '{"groupUid":"acceptance-run-001","memberIndex":0,"enabled":true}'
docker exec -i foretoken-managed-afd-attention /usr/bin/python3 - <<'REQUEST'
import json
import os
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

plan = json.loads(os.environ["FORETOKEN_VLLM_LAUNCH_PLAN"])
artifacts = plan["artifacts"]
tokenizer = AutoTokenizer.from_pretrained(
    artifacts["tokenizer"], revision=artifacts["tokenizerRevision"],
    trust_remote_code=True,
)
request = {
    "request_id": "managed-afd-acceptance",
    "prompt_token_ids": tokenizer.encode("The capital of France is"),
    "sampling_params": {
        "temperature": 0, "seed": 0, "max_tokens": 16,
        "_eos_token_id": tokenizer.eos_token_id,
        "_all_stop_token_ids": [tokenizer.eos_token_id],
    },
}
Path("/tmp/afd-request.json").write_text(json.dumps(request))
with urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:18000/v1/internal/generate",
    data=json.dumps(request).encode(), headers={"Content-Type": "application/json"},
), timeout=600) as response:
    body = response.read()
Path("/tmp/afd-response.ndjson").write_bytes(body)
events = [json.loads(line) for line in body.splitlines()]
assert events and all(event["type"] == "token" for event in events), events
assert events[-1]["finish_reason"] is not None, events
text = tokenizer.decode([token for event in events for token in event["token_ids"]])
assert text.strip(), events
print(text)
REQUEST
docker cp foretoken-managed-afd-attention:/tmp/afd-request.json .
docker cp foretoken-managed-afd-attention:/tmp/afd-response.ndjson .
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  http://127.0.0.1:18001/v1/internal/generate
```

两侧状态须为 `ready: true`，生成须得到非空解码文本，最后的 FFN 请求须返回
404。保存实际 plan、NDJSON、状态、进程 ID、镜像信息、日志与退出码。

延迟 FFN 启动，确认其循环未进入前 Attention 不暴露就绪。分别终止两侧，记录
可见失败，不得在原通信域重启。新建一对，完成请求后先停 Attention，再停 FFN，
检查进程和 GPU 分配无遗留。使用旧控制面实际产生的 V1 重跑普通生成；将 V1/V2
分别交给新旧运行时执行，旧运行时必须在引擎启动前拒绝 V2。本地解析、编译和
CPU 进程检查不代表 GPU、原生传输协议或生成验收通过。

[p2p-source]: https://github.com/vllm-project/afd-plugin/blob/b7561d726b92f366f92634773b731b38e364fbe6/afd_plugin/connectors/gpu/p2p.py
[ffn-source]: https://github.com/vllm-project/afd-plugin/blob/b7561d726b92f366f92634773b731b38e364fbe6/afd_plugin/v1/worker/ffn_worker.py
[worker-source]: https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/worker_base.py
