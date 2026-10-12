<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# Source update lifecycle

English | [简体中文](source-image-lifecycle_zh.md)

Installation and update commands are in [Deploy Foretoken from Source](../custom-deployment.md). The following constraints apply when maintaining the update path.

## Build and publish

Source operations from one workstation are serialized per cluster. Synchronization includes file deletions; engine updates must not restore deleted source from the runtime image. Compilation uses caches separate from model data and preserves compatibility with the selected runtime's Python and accelerator libraries.

A complete application version is published before use and remains immutable. Stop a build's abandoned publishers before reusing its output. The platform owns application storage independently of compiler and model caches.

## Select and recover

Controllers save the selected runtime and application version before creating workloads. Model preparation and serving use the same version; restarts and scaling retain it. Platform installation updates defaults, while explicit redeployment updates existing services.

Deployment success requires the selected version and serving routes to be ready. Existing controllers own rollout and request draining.

## AFD environment and application selection

AFD is one variant of the existing source operation, binding, ClusterBuilder and ApplicationFiles lifecycle. It adds no publisher or cleanup owner. `model-server-afd` contains the same Rust binary and Python adapter export as `model-server`, without ordinary engine overlays. Its internal executable identity remains `model-server`. Mixed Pools share one source revision and publish both directories before submitting that revision.

The chart forwards `runtime.vllm.afd.image` and `applicationURL` to the existing AFD RuntimeProfile and the ModelService application selector. Fresh AFD selections use the AFD directory. Persisted `status.poolApplications`, Pool templates and existing cohorts retain their exact image/URL pair, including legacy ordinary-directory URLs; installation does not silently relocate an old revision. Explicit deployment revision/source revision changes select the current compatible pair. An old source revision must remain published at its selected URL until its consumers and retained history release it.

A prebuilt release supplies matching controller/CRDs, runtime environment and application archive. Source mode retains build inputs, generated image references and per-node digests, and keeps the AFD input image separate from its generated output. AFD V2 must use the matching model-server and adapter; ordinary V1 remains supported by that binary. Installation pairing does not prove old-runtime V2 rejection or GPU protocol compatibility.

The controller init container applies CRDs before the manager starts. Multi-member AFD status requires the role/memberIndex list keys. Never run an older controller against that status or downgrade CRDs while AFD objects remain. Rollback means restoring a recorded compatible chart, CRDs, controller, environment and application combination with retained immutable inputs, not just replacing an image. To leave AFD, first remove its services using the compatible controller and obtain actual process/GPU cleanup evidence; then choose an explicitly compatible earlier combination. Do not force finalizers or assume object disappearance proves release.

Whole-Group retirement/replacement, reliable stop evidence, foreground GC ordering, node-loss classification and some multi-Pool update waits remain predecessor limitations. Automatic recovery and a safety claim for update/delete/rollback remain pending that work and server acceptance. RDMA profiles with multiple members or GPUs per member are rejected; installation must not clear RDMA settings to bypass that rejection. Non-RWX required hostname affinity that conflicts with same-role hard spread also remains rejected.

## Clean up

Keep versions referenced by services, workload templates, running or terminating workloads, saved tasks and Helm rollback history. Remove only the publisher's unreferenced versions, leaving other publishers' work intact.

Helm owns application storage. Source uninstall removes managed compiler caches and the workstation binding while preserving model data.
