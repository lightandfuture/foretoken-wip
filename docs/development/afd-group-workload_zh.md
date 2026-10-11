<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# AFD Group 工作负载生命周期

[English](afd-group-workload.md) | 简体中文

ModelGroup 控制器按 [managed V2 契约](afd-managed-runtime_zh.md) 物化固定 CUDA 1A1F。
创建不代表服务就绪；当前不发布 AFD 生成 Service 或路由，不替换失败 Group。
OCI 和真实 GPU/Kubernetes 验收仍未完成。

## 身份与发现

Pod 名为 `afd-<uid>-attention-0` 和 `afd-<uid>-ffn-0`，UID 来自 API server
持久化的 Group。controller ownerReference 指向 Group，restartPolicy 为 Never。
成员标签为：

- `inference.foretoken.io/model-group-uid`：Group UID；
- `inference.foretoken.io/execution-role`：attention 或 ffn；
- `inference.foretoken.io/member-index`：角色内序号 0。

Group 拥有 headless Service `afd-<uid>-f0`，selector 限定 UID、FFN 角色和序号 0。
`publishNotReadyAddresses: true` 让启动发现不依赖 Ready。两侧 V2 plan 都使用
`afd-<uid>-f0.<namespace>.svc` 和已解析的 connector 端口，不把身份和地址写回 spec。
两侧都提交创建，不等待另一侧 Ready；不引入跨角色反亲和或多成员放置策略。

共享 Pod 模板保留镜像/应用选择、拉取凭据、每侧资源、RDMA、Multus、缓存放置和退出
宽限期。既有 CPU 准备 Job 消费 Attention V2 产物配置，prepare/resolve 不启动引擎。
两侧挂载 Pool 拥有的同一来源记录，peer 入站许可限定当前 Group UID。

## 创建与观测

发现和 Pod 创建前先进入既有 Group drain finalizer。Pod 自创建起携带
`inference.foretoken.io/afd-member`，避免创建成功而 Group status 未保存时对象被删除。
仍由同一 Group 控制器处理该 finalizer，不增加第二生命周期 owner。

`status.afdMembers` 保存角色、实际 Pod UID、锁存的故障原因/消息及
`terminationConfirmed`。直接读取 API server，避免把 informer 延迟当作丢失。
部分创建失败重试复用现有身份，只创建从未创建的角色；不在旧通信域补建失效成员。
故障使 Group 进入 Failed，通过 WorkloadAvailable 报告，Ready 保持 false。
当前 Pool 保留失败 Group，整组替换属于后续恢复实现。

| 原因 | 输入 |
|---|---|
| `MemberFailed` | Failed phase，保留 Evicted 等原始 reason |
| `MemberSucceeded` | 非预期 Succeeded phase |
| `ContainerTerminated` | 必要运行容器终止 |
| `MemberDeleting` | Group 删除流程外的成员删除 |
| `MemberMissing` | 已记录成员对应 Pod 消失 |
| `MemberReplaced` | 固定名称对应不同 Pod UID |
| `NodeUnavailable` | NodeLost、Unknown phase 或未知 readiness |
| `StartupTimedOut` | model-server 运行超过启动期限且 startup probe 尚未成功 |

Pending 与启动超时分开；scheduler condition 继续报告资源不足。不判断“重启耗尽”，
不把 kubelet 重启本身视作终态，不解析 kubectl 展示文本。失效不等于终止。

## 删除与责任边界

Group 删除先正常删除 Attention，让现有运行时执行有界关闭。持久化终止证据后移除成员
finalizer，等待 Pod 消失，再删除 FFN。两个 Pod 都消失后删除 F0 Service，最后释放
Group finalizer。其余 Group 拥有的准备和网络对象沿用垃圾回收；来源记录和缓存保留原 owner。

未调度且正在删除的 Pod 从未运行；已调度 Pod 必须有终态 phase/container 证据，排除
NodeLost、Unknown 和 ContainerStatusUnknown。只有驱逐原因不足以证明终止。
成员消失且没有先前终止证据时保持 `TerminationUnconfirmed`，不强删、不假定 GPU 释放。
节点隔离、整组恢复和路由撤销集成属于后续工作；手动移除 finalizer 会绕过证据边界。

负责人仍须验证同节点通信、独立 GPU、Pending/部分创建、重入、删除、退出和节点失联。
编译与渲染不代表这些场景通过。
