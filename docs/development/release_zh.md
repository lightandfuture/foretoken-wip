<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# 发布版本规则

[English](release.md) | 简体中文

Foretoken 发布 Python distribution、运行环境镜像、应用 `.tar.gz` 文件和 Helm Chart。Python package 遵循 PEP 440，OCI 镜像与 Helm Chart 使用 SemVer。两种格式的具体写法不同，但同一次发布的阶段和序号必须一致。

## 版本阶段

| 阶段 | 用途 | Python 版本 | OCI 与 Helm 版本 |
| --- | --- | --- | --- |
| Development | 可识别的本地或 CI 开发快照 | `0.0.1.dev1` | `0.0.1-dev.1` |
| Alpha | 早期集成与接口验证 | `0.0.1a1` | `0.0.1-alpha.1` |
| Beta | 功能完成后的兼容性与部署验证 | `0.0.1b1` | `0.0.1-beta.1` |
| Release Candidate | 正式发布前的最终验证 | `0.0.1rc1` | `0.0.1-rc.1` |
| Stable | 面向普通安装路径的正式版本 | `0.0.1` | `0.0.1` |
| Python post-release | 修正已发布的 Python 产物或其 metadata | `0.0.1.post1` | 通常复用 `0.0.1`；平台产物变化时发布下一 patch |

同一阶段再次发布时递增末尾序号，例如 `0.0.1a2`、`0.0.1b2` 或 `0.0.1rc2`。只有当前版本达到下一阶段的用途时，才进入下一阶段。

Python 版本的先后顺序如下：

```text
0.0.1.dev1 < 0.0.1a1 < 0.0.1b1 < 0.0.1rc1 < 0.0.1 < 0.0.1.post1
```

Development 版本只用于本地或 CI 快照，不创建 GitHub Release，也不上传 PyPI。OCI 镜像可以额外维护 `latest` 作为日常源码迭代使用的可变别名；`latest` 不是 Helm Chart 版本，也不是发布版本。

## 安装 Python 版本

Alpha、Beta 和 Release Candidate 都是预发布版本。`pip` 默认不会选择这些版本，需要显式允许预发布版本或指定精确版本：

```bash
pip install --pre foretoken
pip install foretoken==0.0.1a1
```

`.postN` 通常复用对应 Stable 版本的平台产物，因为它只修正已发布的 Python package 或 metadata，不承载常规代码变化。如果运行行为或平台产物需要变化，应发布下一 patch，例如 `0.0.2`，而不是把这些变化放进 `.postN`。

## Tag 与版本来源

GitHub Release tag 使用带 `v` 前缀的 Python 版本。

每类产物只有一个权威版本来源：

- `pyproject.toml` 负责 Python distribution 版本。
- `deploy/charts/foretoken/Chart.yaml` 负责 Helm `version` 和 `appVersion`。
- Foretoken OCI 镜像和 Helm Chart package 使用同一发布阶段对应的 SemVer 值。

已经发布的版本不可覆盖。不得重新构建并覆盖 PyPI 或 OCI registry 中已经存在的版本；应根据实际情况递增 Development、预发布、post-release 或 patch 序号。

## Release 描述

创建 GitHub Release 时使用[《Release 描述模板》](release-template_zh.md)。

## 构建与推送发布产物

准备兼容的 NVIDIA 和沐曦推理运行时镜像，然后将下面的仓库前缀和镜像名称替换为实际值：

```bash
export REGISTRY=ghcr.io/your-org/foretoken
export INFERENCE_ENGINE_IMAGE=your-nvidia-runtime:version
export METAX_INFERENCE_ENGINE_IMAGE=your-metax-runtime:version

deploy/release-artifacts build --registry "$REGISTRY"
```

命令构建运行环境镜像，并将 `foretoken-applications-<version>-linux-amd64.tar.gz` 和匹配的 Helm Chart 保存到 `/tmp/foretoken-release`。沐曦镜像 tag 使用 `-metax` 后缀。

只导出应用文件、不重建运行环境时，执行：

```bash
deploy/release-artifacts export --output-dir /tmp/foretoken-release
```

将验证后的压缩包上传为 GitHub Release 附件。下面的 `push` 命令发布运行环境镜像和 Helm Chart，不上传压缩包。

完成产物验证后，登录仓库并推送：

```bash
docker login ghcr.io
helm registry login ghcr.io
deploy/release-artifacts push --registry "$REGISTRY"
```

已有 tag 不会被覆盖，发布中途失败后可使用相同命令重试。完整命令参考见 `deploy/release-artifacts --help`。

## CUDA AFD 发布输入

已有 build/push 组件列表包含 `model-server-afd`，通过 `make image-model-server-afd IMAGE_TARGET=environment` 构建并发布 `model-server-afd-environment:<version>`。打包后的 Chart 选择该环境和归档中的 `model-server-afd/<revision>` 应用来源。归档包含同一源码的 controller、frontend、普通 model-server 和 AFD model-server 目录；AFD 目录复用干净的 Rust/adapter 导出，不包含普通引擎覆盖层。导入按 Chart 实际引用选择组件，因此旧 Chart 仍可导入原三组件归档。

发布 AFD 前，按[镜像指南](afd-runtime_zh.md)将 `VLLM_AFD_BASE_IMAGE` 解析并记录为 Linux amd64 OCI digest，同时记录最终环境 digest、应用归档指纹、源码提交、实际解释器、adapter 路径及测得的依赖版本。[固定来源记录](afd-runtime-sources_zh.md)列出的是候选版本，不是实测支持矩阵。本次源码改动未建立 AFD OCI digest、目标服务器版本或模型 revision，发布填充和服务器验收仍待完成，不能覆盖已有发布版本来增加 AFD。

源码 Chart 的自动版本镜像名只是发布命名约定，不证明 OCI 对象已经存在；对外提供标准安装前须发布完整配套版本。只更新 AFD 镜像不证明 controller/runtime 协议兼容。遵循[保留与回退契约](source-image-lifecycle_zh.md#afd-环境与应用选择)：先部署 CRD 再启动 controller，旧 controller 不能消费多成员 AFD status。

## 发布顺序

1. 确定发布阶段，并按上表更新 `pyproject.toml` 和 `Chart.yaml`。
2. 构建并验证 Python distribution、应用压缩包、Helm Chart 和受影响的 OCI 镜像。
3. 推送对应的 OCI 镜像和 Helm Chart tag。
4. 在实际构建并验证产物的提交上打 tag，发布 GitHub Release 并附上应用压缩包，简述重要改动，并具名感谢贡献者及其提供的支持。
5. 由发布 workflow 将 Python distribution 上传到 PyPI。
6. 验证已发布的 package、镜像、应用压缩包、Chart 和全新安装路径。
