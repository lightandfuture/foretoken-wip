<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# Managed AFD 运行时契约

[English](afd-managed-runtime.md) | 简体中文

model-server 可消费私有 V2 plan，启动本机同步 CUDA 1A1F。当前实现仍待负责人
执行 OCI 构建和 A100 验收。[Group 物化](afd-group-workload_zh.md) 提供固定 Pod 和发现，
服务发布与整组恢复仍待实现。使用运行时源码
提交构建[固定候选镜像](afd-runtime_zh.md)。

## 启动契约与生命周期

普通 Group 的 `FORETOKEN_VLLM_LAUNCH_PLAN` 保留 V1，`nodeCount` 仍表示整个
Group 的节点数。V2 禁止该字段，必须提供 `afd` 中的 `role`、`memberIndex`、
`memberCount`、`groupUid`、`host`、`port`、`numAttentionRanks` 和
`numFfnRanks`。当前仅支持每角色 index=0、一个成员和一个 GPU rank。两侧使用
同一新 Group UID 和指向 FFN 的 rendezvous；多成员输入在引擎启动前明确失败。

模型、生命周期、并行参数和原生参数沿用现有契约。V2 要求
TP=PP=DP=PCP=DCP=1、启用 EP，不支持 KV/EC 传输、KV 事件、EPLB 或原生诊断
工具。Rust 唯一渲染方固定同步 `P2pNcclAFDConnector`、
`compute_gate_on_attention=false`、V1 model runner 和 eager 执行。
其他合法 `additional-config` JSON 内容保留，用户 `afd` 被拒绝，合并后只发
一个参数。原生拓扑、worker class、fault tolerance 和 DBO 覆盖也会拒绝。

适配器随既有 Python 源码包分发，使用 `vllm.general_plugins` 入口；没有私有
状态 socket 环境时不激活。运行时保留显式 `VLLM_PLUGINS` 列表，并加入两个
必需的 AFD 插件。

## 真实就绪与失败

固定插件为 `b7561d726b92f366f92634773b731b38e364fbe6`，Python vLLM 为
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`（0.30.0）。薄适配器委托插件
[P2P connector][p2p-source] 和 [FFN worker][ffn-source] 的真实方法执行。
固定 vLLM 的 [worker 包装层][worker-source] 在构造 worker 之前加载通用插件：

1. `init_afd_connector` 成功后，实际 NCCL 通信组已创建。
2. FFN 原生握手可能早于 busy loop 调用 `start_ffn_server_loop`。
   该方法仅创建线程，不能作为循环就绪。
3. FFN 线程完成 CUDA device 设置，首次进入 `recv_dp_metadata_list` 时，
   在既有 P2P group store 写入 UID 隔离的就绪键，并通过私有 Unix socket
   上报本地 loop 就绪。
4. Attention 等待该键，再上报 connector 完成；这一等待不阻碍独立 FFN
   握手和循环启动。
5. Rust 同时取得原生握手和 worker 事件后才开启 HTTP listener。
   Attention 保留生成入口，FFN 仅有 health、ready 和 `/internal/afd/status`。

状态响应包含 Group UID、角色、成员序号/数量和一个聚合 `ready` 布尔值。
启动阶段没有 HTTP listener，连接失败表示不可用；不伪造分阶段布尔状态。
Kubernetes Group 就绪与路由仍须聚合所有预期成员。

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
