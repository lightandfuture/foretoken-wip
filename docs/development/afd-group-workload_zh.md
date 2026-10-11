<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# AFD Group 工作负载生命周期

[English](afd-group-workload.md) | 简体中文

ModelGroup 控制器按 [managed V2 契约](afd-managed-runtime_zh.md) 物化固定 CUDA 1A1F。
创建不代表服务就绪；同一控制器在完整 AF 就绪后发布仅指向 Attention 的生成
Service 和路由，不替换失败 Group。
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

## 服务就绪与入口撤销

Group 控制器核对完整成员集合、已记录 Pod UID、角色/序号标签、Running/Ready
Pod 及两侧 `/internal/afd/status`。运行时响应把 Group UID 和角色绑定到实际
引擎、connector 和 FFN loop 初始化；端口连通或原生 handshake 不代表完整就绪。

Attention 启动时关闭可恢复的 serving gate。两侧观测全部通过后，控制器向
Attention 的 `/internal/afd/serving` 提交 Group UID、成员序号及 `enabled`，
收到身份匹配、运行时 ready 且 `accepting: true` 的响应，才发布生成 Service
和 Group Ready。FFN 没有此开关或生成 API。该私有契约要求控制器与 model-server
使用匹配构建；缺少 gate 接口的旧运行时保持不可用。

既有 serving snapshot owner 消费 Group Ready，每个 Group 只发布一个指向
Attention Service 的目标。DP/TP 和 KV scope 来自 Attention；GPU 统计包含两侧，
完整 Group 副本仍只计一次。

成员不可用时撤销 Group Ready、删除生成 Service 并关闭 Attention gate，阻止
已有连接上的新请求。各撤销动作在出错时仍独立尝试，由 reconcile 重试。临时观测
失败保持未就绪，完整成员再次通过观测后可以恢复；锁存故障保持 Failed。普通调度
等待和正常删除仍使用各自状态。Ready/WorkloadAvailable 消息标明阻塞角色，
前置 API 错误也会撤销已发布入口。

gate 与运行时 admission 共用原子请求计数，但不能重新打开排空/退出已关闭的
lifecycle admission。在途请求可以完成或失败。观测、Service 删除和前端快照安装
是异步过程，不保证零延迟撤销，也不代表收到前端 generation ACK。Attention 不可达
时无法确认入口关闭。F0 发现保留到既有顺序清理完成。

## 删除与责任边界

Group 删除先撤销 Ready、尝试关闭 Attention gate 并删除生成 Service，再正常删除
Attention，让现有运行时执行有界关闭。持久化终止证据后移除成员
finalizer，等待 Pod 消失，再删除 FFN。两个 Pod 都消失后删除 F0 Service，最后释放
Group finalizer。其余 Group 拥有的准备和网络对象沿用垃圾回收；来源记录和缓存保留原 owner。

未调度且正在删除的 Pod 从未运行；已调度 Pod 必须有终态 phase/container 证据，排除
NodeLost、Unknown 和 ContainerStatusUnknown。只有驱逐原因不足以证明终止。
成员消失且没有先前终止证据时保持 `TerminationUnconfirmed`，不强删、不假定 GPU 释放。
节点隔离和整组恢复属于后续工作；手动移除 finalizer 会绕过证据边界。

负责人仍须验证同节点通信、独立 GPU、Pending/部分创建、重入、删除、退出和节点失联。
编译与渲染不代表这些场景通过。

## Pool 容量与中断续作

Group 由 Pool 创建。持久化的 `modelPoolRef` 与 controller ownerReference
共同确定所属 Pool，`spec.revision` 与 `spec.ordinal` 确定一个容量位置。
Group 名称由 revision 和 ordinal 确定；每次创建由 API server 分配新 UID，
用于成员名称及 F0 通信域。Pool status 只是观测结果，不是创建账本。

revision 选择和容量 reconcile 直接读取 API server 中的子对象。规格匹配的
AFD Group 在删除期间继续保留 revision 并占用原位置，包括 finalizer 等待阶段。
Pool 不会仅因全部匹配 Group 正在删除而分配另一个 revision。对象消失后，
既有容量路径可以补建缺失位置；若已无匹配 cohort，则分配新 revision；若还有
同规格 cohort，则保留其 revision，只补缺失位置。其他 Group 的 UID 和通信域不变。

Group 创建成功可能早于 informer 事件和 Pool status 写入。重入会发现这个已持久化
对象并继续使用其 UID。AFD Create 返回 AlreadyExists 时，直接读回对象，只有
Pool owner 与完整不可变 spec 都匹配才复用。API 错误显式返回，不接管外部对象。
Pool 删除请求携带观测到的 Group UID，防止迟到请求删除同名的新实例。

这些规则只处理容量收敛，不能证明进程已经停止。自动退休失败 Group 尚未接入：
当前 scheduled Pod 判停逻辑可能接受 PodGC 合成的终态 phase，而没有容器退出
证据；foreground GC 也可能独立于 Group 清理循环删除 A/F/F0。节点失联分类和
AFD resolver 的 RDMA 投影仍有缺口。必须先建立可信的 Group 清理边界，自动故障
退休才能安全使用此容量路径；超时或对象删除本身均不证明 GPU 已释放。

AFD Pool 的期望 Group 数为零时，只有直接读取 API 确认所有 revision 均无所属
Group，`RolloutPending` 才变为 false；删除中的 Group 仍保持 pending。空 Pool
不需要 prepared 或 serving revision，`Ready` 仍为 false、reason 为
`ScaledToZero`。此条件只报告对象容量收敛。正常缩容和更新退休仍使用 Group
清理路径，继承上述停止证据及垃圾回收顺序限制；API 空列表不独立证明 GPU 释放。
