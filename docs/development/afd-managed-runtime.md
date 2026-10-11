<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# Managed AFD runtime contract

English | [简体中文](afd-managed-runtime_zh.md)

The model-server consumes private V2 plans for a local, synchronous CUDA 1A1F
pair. This is a runtime implementation, pending owner-executed OCI and A100
acceptance; it does not provide Kubernetes AFD deployment or recovery.
Build the [fixed candidate image](afd-runtime.md) from the runtime source commit.

## Plan and process ownership

`FORETOKEN_VLLM_LAUNCH_PLAN` retains V1 for ordinary groups. V1 requires
`nodeCount` with its existing whole-Group meaning. V2 forbids that field and
requires `afd` with `role`, `memberIndex`, `memberCount`, `groupUid`, `host`,
`port`, `numAttentionRanks`, and `numFfnRanks`. Only index 0, one member and
one GPU rank per role are currently supported. Both roles use the same fresh
Group UID and rendezvous endpoint, which must reach FFN. Multi-member plans
fail before an engine starts.

Shared artifacts, lifecycle, parallelism, and native engine arguments retain
the existing launch schema. V2 requires TP=PP=DP=PCP=DCP=1 and EP enabled,
with no KV/EC transfer, KV events, EPLB, or native diagnostic instrumentation.
The Rust renderer selects synchronous `P2pNcclAFDConnector`,
`compute_gate_on_attention=false`, the V1 model runner and eager execution.
It preserves other `additional-config` JSON entries, rejects user-owned `afd`,
and emits one merged `--additional-config`. Native topology, worker-class,
fault-tolerance and DBO overrides are rejected.

The managed process receives the adapter through the existing Python source
bundle. Its `vllm.general_plugins` entry point is inactive without the private
observation socket environment. The runtime preserves an explicit
`VLLM_PLUGINS` list and includes both required AFD plugins.

## Readiness and failure

The fixed plugin is `b7561d726b92f366f92634773b731b38e364fbe6`; Python vLLM is
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` (0.30.0). The adapter delegates the
actual methods in the [P2P connector][p2p-source] and [FFN worker][ffn-source].
The fixed vLLM [worker wrapper][worker-source] loads general plugins before
constructing the worker:

1. Successful `init_afd_connector` establishes the actual NCCL groups.
2. FFN native EngineCore handshake can finish before its busy loop invokes
   `start_ffn_server_loop`. That method's thread creation is not readiness.
3. The first `recv_dp_metadata_list` entry follows CUDA device setup in the
   actual FFN thread. It publishes a UID-scoped ready key in the existing P2P
   group store, then reports local loop readiness on the private Unix socket.
4. Attention waits for that store key before reporting connector completion.
   This does not prevent the independent FFN handshake and loop from starting.
5. Rust requires native handshake plus the worker events before opening its
   HTTP listener. Attention retains the generation API; FFN has only health,
   readiness and `/internal/afd/status`.

The status response contains Group UID, role, member index/count and one
aggregate `ready` boolean. There is no HTTP status listener during startup;
connection failure must be treated as unavailable. It is not a fabricated
sequence of per-stage status booleans. Kubernetes Group readiness and routing
must still aggregate all expected members.

The startup deadline includes preparation, engine startup and AFD readiness.
The plugin world rendezvous and ready-key wait take their timeout from the
same lifecycle setting; the Rust deadline remains authoritative, including
plugin operations with fixed native timeouts. Fatal FFN loop errors, lost
observation channels, unhealthy EngineCore clients and child exits end this
member. AFD never takes the ordinary temporary-cache restart path. Recovery
requires a new complete Group and communication domain.

Shutdown drains Attention requests, then terminates its managed process group.
Stop Attention before FFN. The upstream Rust patch retains the process group
ID after its launcher has been reaped, respects the caller's graceful deadline,
reaps same-group descendants adopted by PID 1 after Tokio reaps the launcher,
and checks group termination after SIGKILL. Forced cleanup has a five-second
confirmation deadline and reports failure if the group persists.

## Owner acceptance

Use the same image digest, fixed model/tokenizer revision, model cache and two
GPUs described by the image guide. Save the exact two private plans. A minimal
Attention plan is:

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

Save the JSON as `attention.json`, replace both revision placeholders with the
recorded full model commit, and use a new `groupUid` for this attempt. The host
commands require Python 3 and reuse `AFD_IMAGE` and `MODEL_CACHE` from the image
guide. Both containers share the host network, so the loopback rendezvous applies:

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

Follow each container's `docker logs` until startup completes; connection failure
during startup is expected. Then execute:

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

Both status responses must have `ready: true`, generation must produce nonempty
decoded output, and the final FFN request must return 404. Retain the plans,
NDJSON, statuses, process IDs, image metadata, logs and exit codes.

Repeat with delayed FFN startup: Attention must not expose readiness before
FFN enters its loop. Terminate each role separately and record visible failure;
never restart it into the same domain. On a fresh pair, finish requests, stop
Attention, then FFN; check process and GPU allocations for orphans. Repeat
ordinary V1 generation with an actual old-controller plan. Execute that V1
and V2 with both new and old runtime binaries: the old runtime must reject V2
before launching any engine. Local parsing, compilation and CPU process checks
are not these GPU or wire-protocol acceptance results.

[p2p-source]: https://github.com/vllm-project/afd-plugin/blob/b7561d726b92f366f92634773b731b38e364fbe6/afd_plugin/connectors/gpu/p2p.py
[ffn-source]: https://github.com/vllm-project/afd-plugin/blob/b7561d726b92f366f92634773b731b38e364fbe6/afd_plugin/v1/worker/ffn_worker.py
[worker-source]: https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/worker_base.py
