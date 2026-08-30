# ADR-0001：中性采样协议与引擎各自构造

- 状态：已接受（Accepted）
- 日期：2026-08-26
- 修订：2026-08-27（新增方案 B —— frontend 自持中性 IR 作为终态）
- 修订：2026-08-29（方案 B 补充：P0/P1/P3 冻结流式响应契约；conformance mock 采用 golden-sample 基准）
- 范围：`data-plane/model-protocol`、`data-plane/model-server`，最终扩展至 `data-plane/frontend`

## 背景（Context）

Foretoken 从前端（backend 不感知）把 chat/completion 请求路由到 model-server，后者运行单一推理引擎（vLLM 或 SGLang）。两者之间的 wire 契约是 `foretoken-model-protocol` 的 `GenerateInput`/`SamplingParams`。

中性 `SamplingParams` 用类型化字段承载公共采样子集（`temperature`、`top_p`、`top_k`、`max_tokens`、各类 penalty、`stop_token_ids`…），但还带有一个不透明逃生通道：

```rust
pub extra_args: BTreeMap<String, serde_json::Value>
```

### 根因：wire 协议是 vLLM-centric

frontend 内部以 vLLM 建模（`vllm_llm::GenerateRequest` 与 `EngineCoreSamplingParams`）。它的 `llm-facade` 转换（`frontend/src/llm-facade/src/conversion.rs::to_neutral_sampling`）把 `EngineCoreSamplingParams` 的每个 vLLM 专属字段以 **vLLM 的字段名**折叠进 `extra_args`（`all_stop_token_ids`、`structured_outputs`、`logit_bias`、`eos_token_id`、`thinking_token_budget`…）。

因此「中性」的 `extra_args` 实际上是 **vLLM 专属的袋子**。后果：

1. **泄漏**。SGLang 适配器（`model-server/src/engine/sglang/backend.rs`）把 `extra_args` 原样转发进 SGLang 的 `sampling_params`。SGLang 的 `SamplingParams` 拒绝未知关键字，产生 `TypeError: Unexpected keyword argument 'all_stop_token_ids'`，导致每个 SGLang chat 请求都以 `request_failed` 终止。
2. **不对称**。vLLM 是「核心」，拥有袋子并通过白名单消费其 key（`engine/vllm/conversion.rs::to_vllm_sampling`）；SGLang 是「附属」，只能整体忽略袋子。再加第三个引擎会重现同样的问题。

### 参照：Dynamo 的设计

NVIDIA Dynamo 走过同样的轨迹（先 vLLM，再以独立 handler 树硬塞 SGLang/TRT-LLM），随后做了统一后端重构（commit `#8003`），引入中性的 `GenerateRequest.sampling_options`（引擎无关 key），每个引擎适配器自行构造后端参数（`_build_sampling_params`）。每个引擎各有自己的原生 passthrough——vLLM 的 `SamplingParams.extra_args`、SGLang 的 `sampling_params["custom_params"]`——因此引擎保持对称：中性协议从不携带 frontend 填充的引擎专属袋子。

更深的教训是「沙漏」形状：Dynamo 的 frontend 端到端自持中性 IR（`PreprocessedRequest` 携带类型化的 `SamplingOptions`/`StopConditions`/`OutputOptions`），vLLM 类型只出现在后端 adapter 里；其中性 `GuidedDecodingOptions` 是类型化「超集」（同时携带 SGLang 的 `structural_tag` 与 vLLM 的 json/regex/choice/grammar），而非公共子集——子集会再次把引擎专属概念逼进袋子。Foretoken 的方案 B（见下）采用此形状。

## 决策驱动因素（Decision Drivers）

- **零泄漏**：任何引擎都不得收到另一引擎的采样 key。
- **引擎等价**：vLLM 与 SGLang 地位对等，各有类型化的专属扩展通道；没有谁「拥有一个共享袋子」。
- **可扩展**：加引擎 = 加「中性概念 → 后端」映射，不动中性协议、不动其他引擎。
- **稳定性**：不阻塞 SGLang 上线，不破坏已跑通的 vLLM 路径。

## 备选方案（Considered Options）

- **黑名单过滤**（SGLang 适配器过滤掉 vLLM 的 key）。已否决：这是硬编码另一引擎 key 的补丁，依然不对称、难维护。
- **per-engine 枚举**（`enum SamplingExt { Vllm(...), Sglang(...) }`）。已否决作为目标：仍把中性协议与每个引擎的名字耦合，且要求 frontend 知道后端才能正确打标。
- **中性概念 + 引擎各自构造**（选定的目标）。中性协议只承载引擎无关的类型化概念；每个引擎适配器把概念映射到自己的后端，并把引擎专属数据写进自己的原生通道。这正是 Dynamo 的模式。

## 决策（Decision）

本 ADR 之后有两条路线。**方案 A**（S0 + S1/S2）只中性化 wire、落地快——S0 即本次改动，S1/S2 补齐；**方案 B**（frontend 自持中性 IR）是更深的终态，彻底移除 frontend 的 vLLM-centrism，单独排期。

### 本次落地：S0 —— 止住泄漏

SGLang 适配器只从中性 typed 字段构造 SGLang 原生采样参数，**绝不转发 `extra_args`**。引擎专属 key 不是靠过滤丢弃，而是适配器根本不去消费它们。SGLang 自己的原生通道（`custom_params`）由下面的目标设计接入。

风险最小、可解锁 SGLang chat：只改 SGLang 适配器，外加一个「vLLM key 不泄漏」的回归测试。

### 目标（deferred）：方案 A 补齐 —— S1/S2 引擎等价的中性 wire

中性 `SamplingParams` 变成纯引擎无关概念集合：

```rust
pub struct SamplingParams {
    // 现有中性 typed 字段……
    logit_bias: Option<LogitBias>,           // 从 extra_args 提升（中性）
    guided_decoding: Option<GuidedDecoding>, // 中性枚举：Json/Regex/Choice/Grammar
    allowed_token_ids: Option<Vec<u32>>,
    bad_words_token_ids: Option<Vec<u32>>,
    // extra_args 删除
}
```

- 从中性协议中**删除 `extra_args`**。
- 每个引擎适配器把中性概念映射到自己的后端，并在内部拥有后端专属默认值：
  - vLLM：`guided_decoding` → `structured_outputs`；`all_stop_token_ids` 由 `stop_token_ids` + 引擎真实 tokenizer 的 eos 内部计算；其余 vLLM 专属字段（`thinking_token_budget`、`repetition_detection`、`eos_token_id`、`skip_reading_prefix_cache`、`logprob_token_ids`）变为 vLLM 内部实现；引擎专属 passthrough 写入 vLLM `SamplingParams.extra_args`。
  - SGLang：`guided_decoding` → `json_schema`/`regex`/`structural_tag`；引擎专属 passthrough 写入 SGLang `custom_params`。
- **只有当第二个引擎需要某个概念时，才把它提升为中性字段**——这是保证中性协议不累积 vLLM 专属表面积的关键。
- frontend（`frontend/src/llm-facade/src/conversion.rs` 与 `text/src/lower.rs`）停止往协议里注入 vLLM 字段名，改为把 vLLM 的 `EngineCoreSamplingParams` 映射回中性概念——这止住了 wire 上的 vLLM 泄漏，但 frontend 内部仍持有 vLLM 类型（见方案 B）。
- 同类问题：`EngineExtensions`（`mm_features`、`lora_request`、`reasoning_parser_kwargs`）携带 vLLM 类型的不透明值，同样的不对称；并入方案 B 的 P4，而非单独中性化。

### 目标（deferred）：方案 B —— frontend 自持中性 IR

方案 A 只中性化 wire，frontend 内部仍以 vLLM 建模：`foretoken_text` / `foretoken_chat` / `foretoken_engine_core_client` 都是 `pub use vllm_*::*` 的 re-export，frontend 内存里拿的是 `vllm_llm::GenerateRequest` / `EngineCoreSamplingParams`，只在 wire 边缘（`llm-facade/src/conversion.rs`）转成中性协议。这个转换 shim 必须忠实 round-trip 每个 vLLM 字段，vLLM 概念仍会从这里持续渗漏——`extra_args` 正是 shim 的逃生口，每新增一个 vLLM 字段就会再次打开 S0 刚堵住的洞。

方案 B 把中性边界往内推：frontend 端到端自持中性 IR，vLLM 类型只允许出现在两处——

1. **复用点**（tokenizer + chat 模板渲染，产出天然中性的 token ids 与 prompt 文本）；
2. **vLLM adapter**（`data-plane/model-server/src/engine/vllm/`），唯一的中性 → vLLM 转换点。

中性 IR 是类型化「超集」而非公共子集（见上文 Dynamo 参照）。当前 vLLM 类型在 frontend 的渗漏锚点：

- `text/src/lib.rs` / `chat/src/lib.rs` —— `pub use vllm_text::*` / `vllm_chat::*` 暴露 vLLM request processor，其 `prepare()` 产出 `GenerateRequest`。
- `server/src/runtime.rs` —— `RoutedRequest.request: vllm_llm::GenerateRequest`；`prepared.generate_request` 直接流进 router。
- `router/src/request.rs` —— `RouterRequest.generate_request: Arc<vllm_llm::GenerateRequest>`，直接读 `cache_salt` / `lora_request` / `mm_features` / `skip_reading_prefix_cache`。
- `llm-facade/src/{lib.rs,http.rs,facade.rs}` —— `TokenStream` 产出 `GenerateOutput`；`LlmFacade::generate(GenerateRequest)`；PD 编排把 `kv_transfer_params`/`ec_transfer_params` 塞进 `sampling_params.extra_args`。
- `server/src/response.rs` —— decode 流经 `vllm_text::decoded_text_event_stream` 消费 `GenerateOutput`。

分阶段见实施范围；每个切片保持 vLLM 绿灯、可独立 review。

### SGLang 富集成路线图（deferred）

Dynamo 调研结论（其引擎不报告 model dtype；注册信息只有 `context_length`、KV 指标、DP、bootstrap）供后续 SGLang 集成参考：更丰富的 SGLang 元数据、guided decoding、`custom_params` 接入都属于目标设计的下游工作，应建立在它之上。

## 影响（Consequences）

- **正面**：SGLang chat 恢复可用；适配器层已准备好走向「引擎等价」；更深层重构的计划已记录于此，待排期。
- **负面**：原本依赖 SGLang 转发 `extra_args` 的 vLLM-only 请求特性不会被转发（它们对 SGLang 从未生效过）；在 S1/S2 落地前，wire 协议仍携带 vLLM 形状的袋子，SGLang 专属采样扩展暂时无法表达。
- **风险**：S1/S2 触及 frontend（改动量最大）；应作为独立工作排期并单独评审，全程保持 vLLM 路径绿灯。
- **方案 B 风险**：自持 IR 意味着要重写 vLLM 现在免费提供的采样 lowering（P1）与 decode 流（P3）；回归网是现有 vLLM conformance 测试，每个切片都要保持绿灯。

## 实施范围（Implementation Scope）

- **S0（已做）**：`data-plane/model-server/src/engine/sglang/backend.rs` —— 停止转发 `extra_args`；回归测试。
- **S1/S2（未来，方案 A）**：`data-plane/model-protocol/src/types.rs`（中性概念、删除 `extra_args`）；`data-plane/model-server/src/engine/vllm/conversion.rs` 与 `data-plane/model-server/src/engine/sglang/backend.rs`（各自构造）；`data-plane/frontend/src/llm-facade/src/conversion.rs` 与 `data-plane/frontend/src/text/src/lower.rs`（frontend 在 wire 边缘解耦）。
- **方案 B（未来，frontend 自持 IR）**，按依赖序：
  - **P0** —— 冻结中性 IR 类型：类型化 `SamplingParams` 超集、frontend 自有的 `InferenceRequest`/`InferenceOutput`、中性 `disaggregated_params`。只类型 + 签名 + 形状测试。同时冻结流式分帧（一个中性事件 = 一次 token 增量）与对引擎版本漂移的响应形状容错（例如 SGLang 的 finish reason 是对象而非字符串）。
  - **P1** —— 自持采样 lowering：frontend 直接构建中性 `SamplingParams`（默认值/EOS/max-token 解析），不再调 vLLM `TextRequestProcessor::prepare`；vLLM adapter 成为唯一的中性 → `EngineCoreSamplingParams` 映射。依赖 tokenizer 的计算（vLLM 现在免费提供的 `all_stop_token_ids` = stop ids + tokenizer eos）必须显式归属：tokenizer 已是复用点则 frontend 自持，否则归 adapter。
  - **P2** —— 中性请求 envelope：router / llm-facade / server 持有 `InferenceRequest`；删除 `to_generate_input` shim。
  - **P3** —— 中性响应 envelope：`TokenStream` 产出 `InferenceOutput`；用自建的小型 decoder 取代 `decoded_text_event_stream`（tokenizer 仍复用）；`ChatOutputProcessor` 继续解析纯文本。P3 还必须冻结 vLLM 类型目前「隐含但从未声明」、而 SGLang HTTP 流今天正在违反的流式契约：(a) `token_ids` 是逐步**增量**（SGLang 流式返回累积前缀，适配器只发增量）；(b) 首个输出携带 prompt token ids（decoder 用它初始化——SGLang HTTP 流没有 `prompt_info`，适配器必须注入）；(c) terminal chunk 携带 `FinishReason`（SGLang 以对象 `{"type": ...}` 上报，适配器内部归一化为中性枚举）。
  - **P4** —— 收尾：`EngineExtensions` 变为引擎拥有的 opaque 透传（frontend 不再构造 vLLM `MmFeatures`/`LoraRequest`/`ReasoningParserKwargs`）；`kv_transfer_params`/`ec_transfer_params` 更名中性 `disaggregated_params`；re-export crate 收敛到引擎无关面。
- **Conformance mock 基准**：conformance mock 必须以真实引擎响应的 golden sample（抓取的原始响应）为基准，而非手写假设——SGLang 的流式形状（累积 `output_ids`、对象 `finish_reason`）就曾在原始手写 mock 之下漂移，导致「流式 chat 端到端工作」的 commit 与实测不符。
