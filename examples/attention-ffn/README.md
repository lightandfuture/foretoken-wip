<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# DeepSeek-V2-Lite with Attention/FFN disaggregation

English | [简体中文](README_zh.md)

Prepare a DeepSeek-V2-Lite service whose Attention and FFN run in separate GPU
processes. Start with one Attention member and one FFN member on one node
(1A1F), then consider two-node 2A2F. These are **candidate examples, not
deployment-verified configurations or a supported-hardware matrix**.
Runtime validation is deferred to the server at the owner's request and has not
been performed. See [acceptance preparation](../../docs/development/afd-acceptance.md)
for evidence still required, including quality and lifecycle checks.

## Install and prepare the model

Run commands from the repository root on the later validation host. The initial
target is CUDA on two K3s nodes, each with two NVIDIA A100 80 GB GPUs. Start on
one node with two free GPUs; allow CPU, RAM and storage for preparation, the
frontend and the platform in addition to the model resources below.

1. Follow [standard installation](../../cli/README.md#install-the-kubernetes-platform) using a
   **published, matching AFD-capable release**. Its CLI, Chart/CRDs, controller,
   AFD environment image and application archive must have recorded versions
   and fingerprints. Use the existing installation command:

   ```bash
   foretoken install
   ```

   No actual AFD release digest is supplied with this example. An automatic
   image name in Chart values is a naming convention, not proof of publication.
   If the release is unavailable, prepare the fixed source checkout through
   [CUDA AFD source deployment](../../docs/custom-deployment.md#cuda-afd-source-deployment)
   and the [K3s/remote-cluster guide](../../docs/kubernetes-deployment.md). Export
   `VLLM_AFD_BASE_IMAGE` with the owner's recorded CUDA image digest and set `REGISTRY`
   to the registry reachable by that cluster before the existing source command:

   ```bash
   pip install -e .
   foretoken install -e . --registry "$REGISTRY"
   ```

   Retain the same `--values` when overrides are needed. Ordinary engine-source
   overlays do not modify the AFD environment. Install the CRD with
   `status.afdMembers.memberIndex` and role/memberIndex list keys before starting
   the matching controller; the platform installer owns this ordering.

   The fixed source has a reported collision when the explicit image override is
   exactly `runtime.vllm.afd.image: afd-engine`: source builds can interpret that
   OCI name as the internal build stage and omit image handling/provenance. That
   particular path is paused pending its owner's decision; this does not describe
   every image override. No source-build path has runtime validation here.

2. Prepare [shared model storage](../../docs/model-storage.md). The storage
   example requests a 100 GiB RWX PVC from the default StorageClass. Select a
   real RWX class in `storage/cache.yaml` when the default cannot provide it;
   capacity is a starting request, not a measured model/cache requirement.

   ```bash
   kubectl apply -k examples/attention-ffn/storage
   kubectl -n foretoken-afd-demo get runtimecache models -o yaml
   ```

   Have the storage owner populate the claim reported in `status.claimName`
   with a complete **immutable snapshot** of `deepseek-ai/DeepSeek-V2-Lite` at
   `models/deepseek-ai/DeepSeek-V2-Lite` relative to the volume root. Include
   weights, configuration, tokenizer and any required model code. Record the
   actual model and tokenizer commits and file fingerprints before deployment.
   They are owner inputs; this repository does not supply invented revisions.
   Keep the same files available to preparation, frontend and every member.
   Provisioning a PVC does not download or upload these local files.

   The example uses the existing `source: local` contract. Public ModelService
   has no model/tokenizer revision field, and native `revision` overrides are
   platform-owned. Remote `source: hf` resolves a moving provider revision in
   each Pool's preparation record; separate Pools are not thereby proven to
   use identical weights. See [model sources](../../docs/model-sources.md).
   Do not replace the immutable local directory while it is being served.

3. Set `NODE_A` to the approved first node's real name. Ensure **only that node**
   matches `workload-group=afd-single`; the example contains no private hostname.

   ```bash
   kubectl label node "$NODE_A" workload-group=afd-single --overwrite
   kubectl get nodes -l workload-group=afd-single
   ```

## Deploy, request and observe

After the inputs above are ready:

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

This base-model example uses Completions and a plain-text prompt, without an
instruction/chat template. Retain the actual HTTP response and generated token
IDs. `foretoken deploy` and `status` observe controller readiness; they do not
prove a request succeeded or that output quality is acceptable. No successful
response is included here as an acceptance result. Gateway users configure the
frontend hostname through the existing [Gateway guide](../../cli/README.md#gateway-mode).

```bash
foretoken status "$RECIPE" --watch
kubectl -n foretoken-afd-demo get modelservices,modelpools,modelgroups
kubectl -n foretoken-afd-demo get pods -o wide
kubectl -n foretoken-afd-demo get modelgroups -o yaml
```

Inspect every expected member's identity and Ready state, not just the number
of Running Pods. A complete Group must be ready before Attention member zero
provides generation; the remaining Attention members and FFN are not public
generation endpoints. Discovery can exist before serving readiness.

## Choose a layout

All paths share the same namespace and service names: **choose one at a time**.
They are alternatives, not concurrent deployments. Each A/F member requests
one GPU, four CPUs and 48 GiB RAM. BF16 and a 2,048-token context are explicit
example settings, not performance recommendations.

| Path | A members × GPUs/member; DP/TP | F members × GPUs/member; DP/TP | Complete Groups | Total GPUs | Placement prerequisite |
|---|---|---|---:|---:|---|
| `same-node` | 1 × 1; 1/1 | 1 × 1; 1/1 | 1 | 2 | Exactly one eligible node with two free GPUs |
| `two-node` | 2 × 1; 2/1 | 2 × 1; 2/1 | 1 | 4 | Exactly two eligible nodes, each with two free GPUs |
| `two-groups` | 1 × 1; 1/1 per Group | 1 × 1; 1/1 per Group | 2 | 4 | Two eligible nodes; cross-node layout needs controlled capacity |
| `baseline` | Ordinary vLLM, one member, default DP=TP=1 | — | 1 | 1 | One eligible node; same immutable model and BF16 |

For `two-node`, set `NODE_A` and `NODE_B` to the two approved nodes and label
both `workload-group=afd-pair`; ensure no additional node has that value. After
the earlier deployment has actually released its resources, select:

```bash
kubectl label node "$NODE_A" "$NODE_B" workload-group=afd-pair --overwrite
kubectl get nodes -l workload-group=afd-pair
RECIPE=examples/attention-ffn/two-node
foretoken deploy "$RECIPE" --timeout 30m
```

Each role's members have hard hostname separation within their Group. With
exactly two eligible nodes and two GPUs on each, 2A2F places one A and one F
on each node. Cross-role affinity is only a soft preference: it neither
guarantees co-location in general nor pairs A0 with F0 or A1 with F1. Pool
`replicas` counts complete AF instances; it is not role DP or member count.

`two-groups` checks isolation between two independent 1A1F Groups. Merely
deploying its static `replicas: 2` cannot force both Groups across nodes. That
specific check requires the owner to establish exactly one free GPU per node,
start the first Group, then release only preregistered held capacity so the
second Group again sees one free GPU per node. Preserve the first Group's UID
while scaling to two and record actual placement. Do not displace unrelated
workloads or invent role selectors/index pairing. The controlled procedure is
pending [lifecycle acceptance](../../docs/development/afd-acceptance.md).

## Update and delete

For a planned configuration/code change, retain the same immutable inputs,
edit the selected configuration and use the existing explicit redeployment:

```bash
foretoken deploy "$RECIPE" --timeout 30m
foretoken status "$RECIPE"
```

Assess available capacity and the old cohort's retirement before rollout. On
four fully allocated GPUs, 2A2F replacement depends on releasing the old GPUs
before new members can run; that lifecycle remains unverified. Extra capacity
is not an established prerequisite or a substitute for safe retirement. Plan interruption; no
zero-downtime update, transparent recovery or safe old-image rollback is
established. Complete the owner-controlled update/delete checks before using
these commands as a production procedure.

```bash
foretoken delete "$RECIPE" --timeout 30m
```

Deletion targets the frontend and model service. Storage and namespace remain
because they were prepared separately. Keep them until their owner explicitly
retires the data. Object disappearance, a successful CLI exit, a finalizer
removal or `terminationConfirmed` alone does not prove workers stopped or GPUs
were released. Preserve process/GPU evidence before another layout reuses
capacity. Do not force finalizers. Read the existing
[upgrade/rollback boundary](../../docs/development/source-image-lifecycle.md#afd-environment-and-application-selection).

## Constraints and next measurements

- Multi-member persistent storage must be RWX and reachable from every selected
  node. A non-RWX claim's required same-host affinity conflicts with role
  spreading and is rejected. Do not remove cache constraints to bypass it.
- A nonempty RDMA profile combined with multiple members or multiple GPUs per
  member is rejected because the AFD resource projection is incomplete.
  Installation's RDMA discovery does not fix that projection. Do not clear
  RDMA configuration to make the example pass; report this dependency. Even
  1A1F does not establish correct RDMA allocation.
- Pending scheduling, preparation failure and generation failure are distinct.
  Inspect status/events, selected node capacity, PVC mounts and image/application
  identity before changing inputs. Network reachability for peer discovery and
  NCCL must be established in the actual cluster; no transport fallback is promised.
- Reliable stopping, foreground GC order, node-loss classification, failure
  cause preservation, automatic retirement/replacement and some multi-Pool
  updates remain incomplete. Details and paused conclusions are in
  [acceptance preparation](../../docs/development/afd-acceptance.md).

Use the existing [greedy/distribution comparison](../../benchmarks/docs/eval/distribution-comparison.md),
[task quality](../../benchmarks/docs/eval/README.md) and
[performance](../../benchmarks/docs/perf/README.md) commands. The
[AFD measurement preparation](../../docs/development/afd-acceptance.md#quality-and-performance)
records their actual input/output contract and the unresolved request-seed gap.
Quality and TTFT/TPOT are separate evidence; no first-release SLO/SLA is set.
