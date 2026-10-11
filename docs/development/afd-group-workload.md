<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# AFD Group workload lifecycle

English | [简体中文](afd-group-workload_zh.md)

The ModelGroup controller materializes a fixed CUDA 1A1F pair using the
[managed V2 runtime contract](afd-managed-runtime.md). Materialization is not
serving readiness. The same controller publishes an Attention-only generation
Service and route after the complete pair is ready. It does not replace failed
Groups. OCI and GPU/Kubernetes acceptance remain outstanding.

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

## Serving readiness and withdrawal

The Group controller observes the exact member set, recorded Pod UIDs, role/index
labels, Running/Ready Pods and both `/internal/afd/status` responses. The runtime
responses bind the Group UID and role to real engine, connector and FFN-loop
initialization. A listening port or native handshake alone is insufficient.

Attention starts with its reversible serving gate closed. After both members
pass observation, the controller posts the Group UID, member index and `enabled`
to Attention's `/internal/afd/serving`. It requires matching identity, runtime
readiness and `accepting: true` before publishing the generation Service and
Group Ready. FFN exposes neither this gate nor a generation API. This private
contract requires matching controller and model-server builds; an older runtime
without the gate remains unavailable.

The existing serving snapshot owner consumes Group Ready and publishes one Group
target using the Attention Service. DP/TP and KV scope come from Attention; GPU
accounting includes both roles, while replica accounting counts the Group once.

Unavailable members revoke Group Ready, remove the generation Service and close
the Attention gate, including requests on established connections. These steps
are attempted independently on errors and retried by reconciliation. Temporary
observation failure remains unready and can reopen after the complete pair is
observed again; latched member failure remains Failed. Pending scheduling and
normal deletion retain their distinct states. Ready/WorkloadAvailable messages
identify the blocking role. API errors also withdraw existing serving entries.

The gate shares runtime admission's atomic request counter, but cannot undo
lifecycle admission closure during drain/shutdown. Already accepted streams may
finish or fail. Observation, Service removal and frontend snapshot installation
are asynchronous; this is not a zero-delay withdrawal guarantee or a frontend
generation acknowledgement. An unreachable Attention cannot acknowledge closure.
F0 discovery remains until the existing ordered cleanup completes.

## Deletion and responsibility boundaries

Group deletion revokes Ready, attempts to close the Attention gate and removes
the generation Service. It then normally deletes Attention first, invoking its existing bounded
runtime shutdown. After termination is recorded durably, the controller removes
the member finalizer and waits for the Pod to disappear before deleting FFN.
It removes F0 discovery after both Pods disappear, then releases the Group
finalizer. Existing garbage collection removes remaining Group-owned preparation
and network objects. Source records and caches retain their existing owners.

An unscheduled deleting Pod never ran. Scheduled Pods require terminal phase or
container evidence, excluding NodeLost, Unknown and ContainerStatusUnknown.
Eviction alone is insufficient. Missing members without prior termination
evidence retain deletion with `TerminationUnconfirmed`; no forced deletion or
assumed GPU release occurs. Node isolation and whole-Group recovery
remain separate work. Manual finalizer removal bypasses
this evidence boundary.

Owner acceptance must exercise same-node communication, independent GPUs,
Pending/partial creation, re-entry, deletion, exits and node loss. Compilation
and manifest rendering do not establish these results.

## Pool capacity and interrupted reconciliation

The Pool owns Group creation. Its persisted `modelPoolRef` and controller owner
reference identify the Pool; `spec.revision` and `spec.ordinal` identify one
capacity position. The Group name is deterministic for that revision and ordinal.
The API server assigns each creation a new UID, which scopes the member names
and F0 communication domain. Pool status is an observation, not a creation ledger.

Revision selection and capacity reconciliation read children directly from the
API server. A matching AFD Group retains its revision and occupies its position
while deletion is pending, including while its finalizer waits. The Pool does
not allocate a second revision merely because all matching Groups are deleting.
After an object disappears, the existing capacity path can create its missing
position. If no matching cohort remains, the Pool allocates a new revision;
otherwise it retains the surviving cohort and replaces only the missing position.
Other Groups keep their UIDs and communication domains.

A successful Group create may precede both informer delivery and Pool status
publication. Reconciliation discovers that persisted Group and continues with its
UID. An AFD create returning AlreadyExists is read back and accepted only when
the Pool owner and entire immutable spec match. API errors remain errors; retries
do not adopt foreign objects. Pool deletion requests carry the observed Group UID
so a delayed request cannot delete a newer incarnation at the same name.

These rules cover capacity convergence, not proof of physical shutdown. Automatic
failed-Group retirement is not connected: the current scheduled-Pod stop predicate
can accept a PodGC-synthesized terminal phase with no container exit evidence.
Foreground garbage collection can also delete the Group's A/F/F0 dependents
independently of its cleanup loop. Node-loss classification and the AFD resolver's
RDMA projection remain incomplete. A trustworthy Group cleanup boundary must be
established before automatic failure retirement can safely use this capacity path;
neither a timeout nor deletion alone establishes GPU release.
