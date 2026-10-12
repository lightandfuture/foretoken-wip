<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# 源码更新生命周期

[English](source-image-lifecycle.md) | 简体中文

安装和更新命令见[从源码部署 Foretoken](../custom-deployment_zh.md)。维护更新流程时需保持以下约束。

## 构建与发布

同一工作站对同一集群的源码操作串行执行。同步包含文件删除；引擎更新不能从运行时镜像恢复已删除的源码。编译缓存与模型数据分开；构建结果须兼容所选运行时的 Python 和设备库。

完整应用版本发布后才能使用，发布后保持不变。再次使用编译输出前，先停止该构建遗留的发布进程。平台独立管理应用存储，不与编译缓存或模型缓存混用。

## 选择与恢复

控制器先保存选定的运行环境和应用版本，再创建工作负载。模型准备与推理使用同一版本，重启和扩缩容沿用该版本。平台安装更新默认版本，显式重新部署才更新已有服务。

选定版本和服务路由就绪后才能报告部署成功。滚动更新和请求排空由已有控制器负责。

## AFD 环境与应用选择

AFD 是已有 source operation、binding、ClusterBuilder 和 ApplicationFiles 生命周期中的一种选择，不增加发布或清理 owner。`model-server-afd` 包含与 `model-server` 相同的 Rust 二进制和 Python adapter 导出，但不包含普通引擎覆盖层；其内部可执行文件身份仍为 `model-server`。混合 Pool 共用源码 revision，在提交该 revision 前先发布两个目录。

Chart 将 `runtime.vllm.afd.image` 和 `applicationURL` 传给已有 AFD RuntimeProfile 和 ModelService 应用选择器。新 AFD 选择使用 AFD 目录；持久化的 `status.poolApplications`、Pool template 和已有 cohort 保留精确镜像/URL 组合，包括历史普通目录 URL，安装不会静默迁移旧 revision。显式 deployment revision/source revision 变化才选择当前兼容组合。旧源码 revision 在消费者与保留历史释放引用前，必须继续在其选定 URL 发布。

预构建发布提供配套 controller/CRD、运行环境和应用归档。源码模式保存构建输入、生成镜像引用及逐节点 digest，并将 AFD 输入镜像与生成输出分开。AFD V2 必须使用配套 model-server 和 adapter；该二进制继续支持普通 V1。安装配套不证明旧运行时拒绝 V2，也不证明 GPU 协议兼容。

控制器 init container 先应用 CRD，再启动 manager。多成员 AFD status 需要 role/memberIndex 联合列表键；AFD 对象存在时不能运行不理解该 status 的旧 controller，也不能降级 CRD。回退须恢复已记录且兼容的 chart、CRD、controller、环境和应用组合，并保留其不可变输入，不能只换旧镜像。退出 AFD 前，先用兼容 controller 删除相关服务并取得实际进程/GPU 清理证据，再选择明确兼容的旧组合；不能强删 finalizer，也不能把对象消失当作资源已释放。

整组退休/替换、可信停止证据、foreground GC 顺序、节点失联分类及部分多 Pool 更新等待仍有前置缺口。自动恢复以及更新/删除/回退的安全结论，仍依赖这些前置和服务器验收。RDMA profile 非空且多成员或每成员多 GPU 的组合明确拒绝，安装不能通过清空 RDMA 设置绕过；非 RWX required hostname affinity 与同角色硬分散冲突时也继续拒绝。

## 清理

保留服务、工作负载模板、运行中或正在终止的工作负载、已保存任务及 Helm 回滚历史引用的版本。只删除发布者自身不再被引用的版本，保留其他发布者的在途工作。

Helm 管理应用存储。源码卸载删除托管编译缓存和工作站绑定，保留模型数据。
