<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# Managed AFD runtime contract

English | [简体中文](afd-managed-runtime_zh.md)

The model-server consumes private V2 plans for synchronous CUDA AFD roles across
multiple members or GPUs per member. OCI and A100 acceptance remain owner-executed.
[Group materialization](afd-group-workload.md) still supports single-GPU 1A1F only;
multi-member discovery, placement and complete-set publication remain controller
work. Whole-Group recovery is incomplete. Build the [fixed image](afd-runtime.md).

## Plan and process ownership

Ordinary V1 retains nodeCount and LWS environment semantics. V2 forbids nodeCount;
afd supplies role, memberIndex, memberCount, groupUid, shared F0 host/port,
numAttentionRanks and numFfnRanks. Both roles share one fresh Group UID and a
rendezvous targeting only the member hosting F0. A multi-member role also requires
an afd.coordination object with host, masterPort and handshakePort.

The Group materializer supplies this UID-isolated, readiness-independent endpoint
for member zero of each role. masterPort coordinates native workers; handshakePort
connects that role's EngineCores to its Rust leader. Shared afd.host/port independently
locates cross-role F0. Runtime consumes endpoints and creates no discovery objects.
The materializer reserves nonconflicting ports and permits native peer traffic,
including dynamic transport ports. Roles sharing a network namespace need distinct
role ports. Neither endpoint may balance across other members or Groups.
Single-member plans omit coordination and retain native local coordination.

BuildAFDLaunchPlan is the actual Go projection boundary. The Pool compiler and
Group workload owner still reject multi-member and multi-GPU materialization.
Direct projection does not establish ModelService-to-multiple-Pod deployment.
No public API or CRD field is added.

Each role satisfies memberCount × GPUs per member = DP × TP. Local GPUs must divide
TP or be divisible by TP, matching the fixed native homogeneous placement. Attention
ranks must be at least FFN ranks; an integral cross-role ratio is not required.
V2 retains PP=PCP=DCP=1, EP enabled, no KV/EC transfer, KV events, EPLB or native
diagnostics. Rust alone renders synchronous P2pNcclAFDConnector, async_dp=false,
compute_gate_on_attention=false, V1 model runner and eager execution. It preserves
other additional-config content, rejects user afd and native topology overrides,
and emits one merged JSON argument. Native aliases and abbreviations cannot
supplant platform topology.

| Identity | Producer and consumer |
|---|---|
| Role member index/count | Materializer → V2 → node-rank/nnodes; no physical-node deduplication |
| GPUs per member G | Normalized resources; Rust derives DP × TP / memberCount |
| DP placement | Fixed vLLM: local DP=max(G / TP, 1), first global DP rank=floor(memberIndex × G / TP) |
| TP rank | Native worker process groups; TP may span members |
| AFD role rank | Fixed plugin: global DP rank × TP + TP rank (PCP=1) |
| AFD world rank | FFN: role rank; Attention: numFfnRanks + role rank |
| Group UID | Persisted Group → plans, discovery isolation and readiness keys |

Rust checks the exact local interval [memberIndex × G, (memberIndex + 1) × G).
A Pod ordinal does not identify all its workers. Fixed vLLM argument handling owns
DP placement; plugin topology owns AFD rank mapping. V2 member identity comes from
the plan and its routable multi-member Pod address from FORETOKEN_MEMBER_IP.
V1 retains LWS_WORKER_INDEX, LWS_LEADER_ADDRESS and its Group UID environment.
AFD does not run the ordinary MetaX calibration launcher.

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

1. Every worker completes actual connector initialization.
2. Each FFN rank enters recv_dp_metadata_list after selecting its CUDA device,
   writes a UID/rank key to its existing AFD subgroup store and publishes loop_ready.
   Thread creation alone is insufficient. FFN handshake may precede loop startup.
3. Each Attention worker waits for its mapped FFN subgroup key before reporting
   connector_ready; this also covers workers outside the smaller metadata group.
4. Attention publishes worker_ready after native compile/warmup completes.
5. Rust requires connector and execution events from every expected local role rank,
   retaining one channel per rank. Leaders also handshake with every DP EngineCore;
   native executors initialize and monitor TP workers, including remote workers.

The existing status response contains Group UID, role, member index/count and
aggregate member readiness. HTTP status is served only after startup; failed
connections and timeouts mean unavailable. A member's readiness never substitutes
for the complete Group set. Only Attention member zero exposes generation and the
initially closed serving gate. Other Attention members and every FFN member expose
lifecycle/status only. The Group owner must observe the complete set before opening
admission; see [serving readiness](afd-group-workload.md#serving-readiness-and-withdrawal).
No new state or recovery protocol is introduced.

Prepare/resolve retain the shared artifact/cache lifecycle and never start engines.
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
```

After verifying both responses against the saved plans and confirming both are
ready, the owner of this standalone acceptance pair opens Attention admission
explicitly (the Kubernetes Group controller performs this step in a deployment):
Use the `groupUid` from the saved plans if it differs from the example.

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
