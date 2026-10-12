<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# AFD candidate scope and acceptance preparation

English | [简体中文](afd-acceptance_zh.md) · [Deployment example](../../examples/attention-ffn/README.md)

This record prepares owner-executed acceptance of DeepSeek-V2-Lite AFD. It is
**not a verified support matrix or first-release acceptance**. As of 2026-10-12,
runtime validation is deferred to the server at the owner's request and has
not been performed. Local source self-review is neither independent review
nor execution evidence. Final review must cover the integrated result tree.

## Candidate scope and evidence

The source reference for these preparations is commit
`e0a9f8331110a65bd03f7a50f3c211e18a5a0ad1`, tree
`cec217a80b3ba76f4850260fec50f8afbbcb1ec9`. The local example/documentation
commit adds no runtime or controller fixes to it.

| Area | Plan / source implementation | Evidence state |
|---|---|---|
| Initial hardware/model | CUDA, DeepSeek-V2-Lite, two K3s nodes with two A100 80 GB each; start 1A1F and deliver 2A2F | Target only; actual server versions, model/tokenizer commits and deployment evidence missing |
| Runtime | [Pinned plugin and Python vLLM](afd-runtime-sources.md), Go V2 projection and Rust-owned argv; ordinary V1 retained | Candidate implementation; predecessor source reviews do not establish GPU or old-runtime compatibility |
| Group scheduling and serving | [UID/role/memberIndex identities, complete Ready set, A0 generation, A0/F0 coordination and shared F0](afd-group-workload.md) | PR-09 independently source-reviewed; three layouts not run |
| Installation / publication | [Separate AFD environment and application](source-image-lifecycle.md#afd-environment-and-application-selection) through the existing installer | PR-10 independent source review archived: PR10-F1/P2 blocks one explicit-image path; E1–E7 evidence gaps remain; no release digest or installation evidence |
| Examples | Shared local-model input; 1A1F, 2A2F, two complete 1A1F Groups and ordinary baseline | Local preparation; no Kustomize rendering or actual deployment evidence |
| Quality / performance | Existing eval/perf consumers described below | Not run; seed and AFD environment-snapshot gaps remain |
| Recovery / update / deletion / rollback | Existing lifecycle owners with the dependencies below | Safety conclusions paused; no transparent recovery or interruption-free claim |

Model registration and accepted topology arithmetic do not establish deployment
support. Record each later case as not run, passed, failed or dependency-paused,
with its source and evidence. Never fill missing results with success values.

## Ownership and fixed inputs

The CLI/Chart choose a matching environment and application; ModelService owns
public intent. The compiler and resolver produce Pool/Group contracts; preparation
owns model acquisition and source records; Group reconciliation owns members,
discovery, serving publication and cleanup. Rust renders the private plan and
owns engine execution. Benchmark/eval consume the existing service and output
interfaces. They do not become another deployment controller or evaluator.

Before server work, the owner records:

- Final source commit/tree, matching CLI/Chart/CRD/controller versions, AFD
  environment OCI digest, application archive fingerprint, ordinary baseline
  engine digest and measured Python/vLLM/plugin/CUDA/NCCL versions. Follow the
  [release owner](release.md#cuda-afd-release-inputs); image naming is not publication.
- Actual K3s/Kubernetes, driver, GPU and node inventory; selector membership,
  available capacity, network reachability and any RDMA profile. Preserve the
  profile when recording a rejected combination; do not disable it as a workaround.
- Identical DeepSeek-V2-Lite model and tokenizer revisions, config/model-code
  fingerprints, BF16 (or identically selected FP16), storage source and mounts.
  The example uses immutable local files because the public API does not expose
  an explicit provider revision. Record those files separately: the normalized
  revision `local` is not a content fingerprint. Controller-owned remote
  `source.json` fixes one Pool revision, not every independent baseline/candidate.
- Input corpus/task revision and hashes, actual prompt tokens or chat template,
  decoding parameters, seed, client dependencies and load settings. Engine and
  hardware versions may differ but must be recorded for both sides.

## Server sequence

All steps below are **preparation, not commands executed in this delivery**.

1. Place the final fixed code on the validation host. Run the applicable repository,
   Go/Rust, Helm and Kustomize checks deferred from local development. Use their
   existing owning commands, not a new validation driver or permanent CI gate.
   Confirm CRDs with role/memberIndex list keys precede the matching controller.
2. Follow the [example's shortest path](../../examples/attention-ffn/README.md):
   standard release or recorded source install, provision/populate immutable
   shared storage, select one node, deploy 1A1F, request, and inspect status.
   Retain the rendered public input, selected image/application identities,
   complete Group/member/Pod UIDs, actual nodes, HTTP response and token IDs.
3. For 2A2F, establish exactly two eligible nodes with two free GPUs each. Record
   two A and two F members, role DP=2/TP=1, one GPU per member, total four GPUs.
   Verify same-role spread, every runtime Ready identity and only A0 generation.
   Verify role coordination and shared F0 are distinct and scoped to the same UID;
   do not infer A/F index pairing from co-location.
4. For two independent 1A1F Groups sharing two nodes, control capacity before
   scheduling each Group: exactly one free GPU on each node. Use only owner-approved,
   preregistered capacity holders; release the first holders only after Group one
   is placed, then add Group two. Change only desired replicas while retaining
   the first UID. A new explicit deploy also selects current applications, so
   do not assume it is a scale-only operation. Retain UID/role/index/Pod identity,
   discovery selectors, actual placement and independent requests. Static
   `replicas: 2` cannot guarantee this cross-node layout. This controlled check
   remains pending capacity and lifecycle decisions; no holder script is supplied.
5. Measure quality and performance independently using the existing entries
   below after resolving their blockers. Record failures as failures, including
   invalid comparison samples. The owner supplies discrepancy analysis and any
   acceptance rationale; performance improvement cannot replace quality acceptance.
6. In the owner-controlled lifecycle phase, capture request admission/withdrawal,
   in-flight completion or error, actual process groups/children and GPU allocation
   through update, normal deletion, startup failure, member fatal, node loss and
   foreground deletion. Record before/after UIDs and capacity. Object removal,
   timeout, a signal or termination flag alone is insufficient stop evidence.
   Do not promise recovery or rollback until the owning defects and evidence close.

## Quality and performance

Use the existing [comparison guide](../../benchmarks/docs/eval/distribution-comparison.md)
and [task guide](../../benchmarks/docs/eval/README.md). No second evaluator is added.
The current consumption contract is:

| Stage | Producer → consumer | Actual behavior / result |
|---|---|---|
| Dispatch | `evaluation.py` → `config/evaluation.py` → `config/distribution_comparison.py` | `--reference`/`--reference-url` selects comparison; `--greedy-compare` selects generated sequences; no comparison `--seed` option |
| Input | `runs/distribution_comparison.py` | Ordered raw text/JSONL text windows, tokenizer BOS if defined; same token-ID prefixes sent to both sides; no chat template applied |
| Request | `integrations/distributions.py`, `CompletionComparisonClient.generate` | Completions request with token-ID prompt, max_tokens, temperature=0, stream=false, return_token_ids=true; **no seed sent** |
| Response | Frontend Completions output → comparison client | Requires native `choices[0].token_ids`, text and normal stop/length finish_reason; no re-tokenization of generated text |
| Sequence result | Run → `results/distribution_comparison.py` | `greedy_comparison_candidates.csv`: valid_samples, failed_samples, exact_matches, exact_match_rate; `greedy_comparison_samples.csv`: token IDs, finish reasons, mismatch_count, zero-based first_divergence (length mismatch uses shorter length); errors excluded from valid-pair rate |
| Other outputs | Existing quality and distribution publishers | `metrics.json` includes prompt_windows/protocol; `serving_environments.json` records available environment; task runs retain metrics.json/native/evaluator.log; distribution produces candidate CSV and positions JSONL |

The default distribution comparison is teacher forcing with full-vocabulary
logprobs, not greedy continuation or task quality. It requires `max-logprobs: -1`
on compatible engines; the minimal example does not enable it. Greedy comparisons
do not support `--resume`. Temperature zero does not supply the missing same-seed
evidence; raw text on both sides is a deliberate common input, not proof of
different prompts.

Two concrete dependencies limit the prepared commands:

1. **Seed:** comparison configuration has no seed field, its request producer
   omits seed and its protocol output cannot prove a common request seed. The
   frontend accepts seed, but that does not repair this consumer. Strict seeded
   acceptance remains pending a decision/fix by the benchmark owner.
2. **AFD environment snapshot:** `results/environment.py:serving_environment`
   indexes `group.spec.resources` and `parallelism`. AFD Group producers omit
   those fields and store role resources/topology under `spec.afd`. The consumer
   can raise an uncaught KeyError for a Kustomize AFD target before evaluation.
   Its error boundary catches DeploymentError only. The affected eval/perf path
   remains dependency-paused; this documentation branch does not fix it.

An existing `--url` source does not touch Kubernetes or that Group snapshot.
It remains an **unexecuted preparation alternative**, requiring manual environment
evidence and two separately identified running endpoints. It does not resolve
the seed gap. Set `BASELINE_URL` and `AFD_URL` to their real Completions URLs,
`TOKENIZER_DIR` to the client-local pinned tokenizer/config directory, and
`CORPUS` to the owner-selected text/JSONL corpus before this example command:

```bash
foretoken eval --url "$AFD_URL" --model deepseek-ai/DeepSeek-V2-Lite \
  --reference-url "$BASELINE_URL" --reference-model deepseek-ai/DeepSeek-V2-Lite \
  --tokenizer-path "$TOKENIZER_DIR" --dataset "$CORPUS" \
  --greedy-compare --context-length 128 --num-windows 4 --max-tokens 64 \
  --output local
```

These sample sizes are preparation values, not an acceptance threshold. URL mode
cannot invent Gateway Host headers; use a correctly routable hostname/HTTPS URL.
URL environment output has no authoritative engine or hardware inventory; attach
the source/digest/topology records separately. The example variants share names,
so passing their directories as if they were distinct live services can reuse
the same existing deployment unchanged. Do not compare an AFD endpoint with
itself. Sequential switching also depends on trustworthy resource release;
four GPUs cannot host 2A2F plus the ordinary baseline simultaneously.

For task scores, the owner selects identical evaluator/task versions, sample
IDs, templates, decoding and seed settings using existing native evaluator
options. Record the full resolved configuration and requests, then retain
`metrics.json`, `native/` and `evaluator.log`. For example, `foretoken eval --url`
with `--evaluator lm-eval --tasks piqa --limit 100 --output local` uses the
existing likelihood path; supply the actual model and client tokenizer through
the [task guide](../../benchmarks/docs/eval/README.md#candidate-likelihood-and-perplexity).
Native task RNG settings alone do not prove a common service request seed.

For TTFT/TPOT, run the existing streaming performance entry against each actual
endpoint with the same prompt/load. Set `SERVICE_URL` to that endpoint's Chat
Completions URL and `PROMPT` to the fixed owner-approved text:

```bash
foretoken perf --url "$SERVICE_URL" --model deepseek-ai/DeepSeek-V2-Lite \
  --prompt "$PROMPT" --num-prompts 20 --max-concurrency 1 --max-tokens 64 \
  --temperature 0 --extra-body '{"seed":0}' --stream --output local
```

The base-model example supplies no verified chat template. The owner must first
provide and confirm the same template on both endpoints before this chat-based
performance command can be used. That prerequisite is pending; this workload
differs from the raw-token greedy protocol. Retain successes/errors and
[TTFT/TPOT definitions](../../benchmarks/docs/metrics.md), input/output lengths,
load and hardware. No SLO/SLA or improvement claim follows from these commands.

## Paused lifecycle conclusions

PR-10's independent review has reported a separate source-build image collision:
explicit `runtime.vllm.afd.image: afd-engine` can be confused with the internal
stage produced from `auto`, skipping local-image handling and provenance.
The full independent report is archived with PR10-F1/P2 as its sole confirmed
finding; the second round adds no separate finding. E1–E7 remain evidence gaps
for the full installation/publication chain, and final integrated-tree review
(G11) remains incomplete. Only that explicit-name path is paused;
other overrides are not implicated by this finding or thereby validated. No
predecessor repair is included in this documentation delivery.

| Dependency | Trigger and impact | Evidence still needed |
|---|---|---|
| EX-003/F1 | PodGC terminal phase with empty/Waiting container state can become terminationConfirmed and release member finalizers without trustworthy physical stop | Owning stop-evidence fix; process/child/GPU observations |
| F2 | Foreground GC of Group controller dependents can bypass intended A → F → discovery order | Owning cleanup fix and real foreground deletion evidence |
| F3 | Node loss can be classified as MemberDeleting instead of node unavailability | Node-loss classification and no false termination inference |
| EX-004 | Non-Failed observation root cause may be overwritten by ServingReconcileFailed | Preserve role/member cause through withdrawal |
| Group recovery | Failed same-spec Group can remain retained; automatic retirement/full replacement incomplete; W1/W2 not run | Trusted stop → retirement → new UID replacement, measured execution |
| Multi-Pool update | A changed Pool scaled to zero may retain old serving capacity while another Pool waits for its GPUs | Cross-Pool capacity/commit progress and actual release |
| RDMA/cache | Nonempty RDMA plus expanded topology is rejected; non-RWX required hostname affinity conflicts with multi-member spread | Owning projection work and compatible real storage/network configuration |

Repair `1776fe081373e6d4694370100b01ec077ba29b89` is not an ancestor of the
fixed source above. Its ownerReference layout is not applied or claimed here.
Reports of source review do not close these dependencies. Keep user-visible
recovery, exit, update and rollback safety conclusions paused while proceeding
with independent documentation and example preparation.

## Documentation references

Read on 2026-10-12: [KServe 0.20 LLM deployment](https://kserve.github.io/website/docs/getting-started/genai-first-isvc)
and [Helm 4.3.0 quickstart](https://helm.sh/docs/intro/quickstart/).
They informed the reader order: prerequisites and installed platform, minimal
service, real request, status, cleanup; alternatives link to their installation
owner. Their APIs, infrastructure and acceptance claims are not Foretoken inputs.
