<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# AFD Group workload lifecycle

English | [简体中文](afd-group-workload_zh.md)

The ModelGroup controller materializes a fixed CUDA 1A1F pair using the
[managed V2 runtime contract](afd-managed-runtime.md). Materialization is not
serving readiness. It publishes no AFD generation Service or route and does
not replace failed Groups. OCI and GPU/Kubernetes acceptance remain outstanding.

## Identity and discovery

The API server's Group UID scopes Pod names `afd-<uid>-attention-0` and
`afd-<uid>-ffn-0`. Both have a Group controller owner reference and
`restartPolicy: Never`. Member labels are:

- `inference.foretoken.io/model-group-uid`: Group UID;
- `inference.foretoken.io/execution-role`: attention or ffn;
- `inference.foretoken.io/member-index`: role-local index 0.

The Group owns headless Service `afd-<uid>-f0`, selecting UID, FFN role and
index 0. `publishNotReadyAddresses: true` separates startup discovery from
readiness. Both V2 plans use `afd-<uid>-f0.<namespace>.svc` and the resolved
connector port; identity and address never enter spec. Both members are
submitted without waiting for peer readiness. No cross-role anti-affinity or
multi-member placement policy is introduced.

The shared Pod-template builder preserves image/application selection, pull
credentials, role resources, RDMA, Multus, cache placement and shutdown grace.
The existing CPU-only preparation Job consumes an Attention V2 artifact plan;
prepare/resolve does not start an engine. Both members mount the Pool-owned
source record. Peer ingress is scoped to this Group UID.

## Creation and observation

The existing Group drain finalizer precedes discovery and Pod creation. Pods
carry `inference.foretoken.io/afd-member` from creation, preventing deletion
between successful creation and Group status persistence. The same Group
controller handles this finalizer; there is no second lifecycle owner.

`status.afdMembers` retains role, actual Pod UID, latched failure reason/message
and `terminationConfirmed`. Direct API-server reads avoid informer lag being
treated as member loss. Partial creation retries reuse existing identities;
only never-created roles can be created. Failed members are never replaced in
the same communication domain. Failure sets Group phase Failed and appears in
WorkloadAvailable; Ready remains false. The Pool currently retains failed
Groups. Whole-Group replacement belongs to the recovery implementation.

| Reason | Observation |
|---|---|
| `MemberFailed` | Failed phase, with original reason such as Evicted |
| `MemberSucceeded` | Unexpected Succeeded phase |
| `ContainerTerminated` | Required runtime container termination |
| `MemberDeleting` | Member deletion outside Group deletion |
| `MemberMissing` | Recorded member Pod disappeared |
| `MemberReplaced` | Fixed name contains a different Pod UID |
| `NodeUnavailable` | NodeLost, Unknown phase or unknown Pod readiness |
| `StartupTimedOut` | Running model-server exceeds startup duration before startup probe succeeds |

Pending scheduling remains separate from startup timeout. Scheduler conditions
still report insufficient capacity. Neither restart exhaustion, kubelet restart
alone nor kubectl display strings define failure. Failure is not termination.

## Deletion and responsibility boundaries

Group deletion normally deletes Attention first, invoking its existing bounded
runtime shutdown. After termination is recorded durably, the controller removes
the member finalizer and waits for the Pod to disappear before deleting FFN.
It removes F0 discovery after both Pods disappear, then releases the Group
finalizer. Existing garbage collection removes remaining Group-owned preparation
and network objects. Source records and caches retain their existing owners.

An unscheduled deleting Pod never ran. Scheduled Pods require terminal phase or
container evidence, excluding NodeLost, Unknown and ContainerStatusUnknown.
Eviction alone is insufficient. Missing members without prior termination
evidence retain deletion with `TerminationUnconfirmed`; no forced deletion or
assumed GPU release occurs. Node isolation, whole-Group recovery and routing
withdrawal integration remain separate work. Manual finalizer removal bypasses
this evidence boundary.

Owner acceptance must exercise same-node communication, independent GPUs,
Pending/partial creation, re-entry, deletion, exits and node loss. Compilation
and manifest rendering do not establish these results.
