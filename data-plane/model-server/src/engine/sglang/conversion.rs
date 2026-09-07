// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! SGLang request/response conversion.
//!
//! Translates vLLM's request and response types to SGLang's native
//! `/generate` wire: [`SglangRequest`] is built from a vLLM `GenerateRequest`
//! via `TryFrom`, and [`SglangResponseDecoder`] turns streamed
//! [`SglangChunk`]s into per-step output fields. Sampling params SGLang
//! cannot honor are rejected before they leave this module.

use std::collections::BTreeMap;

use serde::Deserialize;
use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs, TokenLogprob};
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use vllm_llm::{FinishReason, GenerateRequest};

/// Request body for SGLang's native `/generate`.
#[derive(serde::Serialize)]
pub(super) struct SglangRequest {
    input_ids: Vec<u32>,
    sampling_params: serde_json::Value,
    stream: bool,
    /// Request-level logprob knobs; omitted unless `logprobs` was set.
    #[serde(skip_serializing_if = "Option::is_none")]
    return_logprob: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs_num: Option<u32>,
    /// Backend-native fields merged into the `/generate` body as top-level
    /// keys (e.g. `custom_params`, `session_params`).
    #[serde(flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

impl TryFrom<&GenerateRequest> for SglangRequest {
    type Error = &'static str;

    fn try_from(request: &GenerateRequest) -> Result<Self, Self::Error> {
        if let Some(field) = find_rejected_field(&request.sampling_params) {
            return Err(field);
        }
        if let Some(field) = find_reserved_extension_key(&request.extensions) {
            return Err(field);
        }
        let (return_logprob, top_logprobs_num) = to_sglang_logprobs(&request.sampling_params);
        Ok(SglangRequest {
            input_ids: request.prompt_token_ids.clone(),
            sampling_params: to_sglang_sampling(&request.sampling_params),
            stream: true,
            return_logprob,
            top_logprobs_num,
            extensions: request.extensions.clone(),
        })
    }
}

/// SGLang logprob triple: `[logprob, token_id, token_text]`. Token text is
/// null unless the request set `return_text_in_logprobs`.
type LogprobTriple = (f32, u32, Option<String>);

/// One streamed token chunk from SGLang.
#[derive(Debug, Deserialize)]
pub(super) struct SglangChunk {
    output_ids: Vec<u32>,
    #[serde(default)]
    meta_info: Option<SglangChunkMeta>,
}

#[derive(Debug, Deserialize)]
struct SglangChunkMeta {
    /// SGLang reports finish reasons as objects (`{"type":"stop",...}`,
    /// `{"type":"length","length":N}`, ...), so keep the raw value and
    /// extract the reason kind from its `type` field.
    #[serde(default)]
    finish_reason: Option<serde_json::Value>,
    /// Cumulative per-generated-token logprobs, aligned with `output_ids`;
    /// empty when the request did not ask for logprobs.
    #[serde(default)]
    output_token_logprobs: Vec<LogprobTriple>,
    /// Cumulative top-k alternatives per generated token.
    #[serde(default)]
    output_top_logprobs: Vec<Vec<LogprobTriple>>,
}

/// Parses one line of SGLang's streaming `/generate` response (Server-Sent
/// Events) into a chunk.
///
/// Each payload line looks like `data: {json}`. Lines that carry no payload
/// (blank lines, comment/heartbeat lines, and the `[DONE]` terminator) yield
/// `Ok(None)`; a malformed `data:` payload yields `Err(())`.
pub(super) fn parse_sse_chunk(line: &[u8]) -> Result<Option<SglangChunk>, ()> {
    let line = std::str::from_utf8(line).map_err(|_| ())?.trim();
    if line.is_empty() || line.starts_with(':') || line == "[DONE]" {
        return Ok(None);
    }
    let line = line.strip_prefix("data:").unwrap_or(line).trim();
    serde_json::from_str(line).map(Some).map_err(|_| ())
}

/// Maps an SGLang finish-reason object into vLLM's [`FinishReason`].
fn parse_sglang_finish_reason(reason: &serde_json::Value) -> FinishReason {
    match reason.get("type").and_then(|kind| kind.as_str()) {
        Some("stop") => FinishReason::Stop(None),
        Some("length") => FinishReason::Length,
        Some("abort") => FinishReason::Abort,
        Some("repetition") => FinishReason::Repetition(None),
        _ => FinishReason::Error,
    }
}

/// Windows one chunk's cumulative logprobs to its token delta and converts
/// them into vLLM's per-output [`Logprobs`].
///
/// SGLang streams these arrays cumulatively, aligned with the cumulative
/// `output_ids`, so the same previous-length cut applies. SGLang reports no
/// vocab ranks; entries use their 1-based position in the returned list.
fn chunk_logprobs(
    best: &[LogprobTriple],
    top: &[Vec<LogprobTriple>],
    start: usize,
    end: usize,
) -> Option<Logprobs> {
    let window = best.get(start..end.min(best.len()))?;
    if window.is_empty() {
        return None;
    }
    let positions = window
        .iter()
        .enumerate()
        .map(|(offset, &(logprob, token_id, _))| {
            let mut entries = vec![TokenLogprob {
                token_id,
                logprob,
                rank: 1,
            }];
            // The top-k list may repeat the chosen token; vLLM expects
            // alternatives only.
            if let Some(runners) = top.get(start + offset) {
                for &(runner_logprob, runner_id, _) in runners {
                    if runner_id == token_id {
                        continue;
                    }
                    entries.push(TokenLogprob {
                        token_id: runner_id,
                        logprob: runner_logprob,
                        rank: entries.len() as u32 + 1,
                    });
                }
            }
            PositionLogprobs { entries }
        })
        .collect();
    Some(Logprobs { positions })
}

/// The SGLang-derived fields of one decoded output step.
#[derive(Debug)]
pub(super) struct DecodedStep {
    pub(super) token_ids: Vec<u32>,
    pub(super) logprobs: Option<Logprobs>,
    pub(super) finish_reason: Option<FinishReason>,
}

/// Decodes SGLang's streamed chunks into per-step vLLM output fields.
///
/// SGLang streams cumulative `output_ids` (and cumulative logprobs), so the
/// decoder tracks how much has been emitted and cuts each chunk down to its
/// delta.
#[derive(Default)]
pub(super) struct SglangResponseDecoder {
    previous_output_len: usize,
}

impl SglangResponseDecoder {
    pub(super) fn decode(&mut self, chunk: SglangChunk) -> DecodedStep {
        let output_len = chunk.output_ids.len();
        let token_ids = chunk
            .output_ids
            .get(self.previous_output_len..)
            .unwrap_or(&[])
            .to_vec();
        let finish_reason = chunk
            .meta_info
            .as_ref()
            .and_then(|meta| meta.finish_reason.as_ref())
            .map(parse_sglang_finish_reason)
            .or_else(|| token_ids.is_empty().then_some(FinishReason::Length));
        let logprobs = chunk.meta_info.as_ref().and_then(|meta| {
            chunk_logprobs(
                &meta.output_token_logprobs,
                &meta.output_top_logprobs,
                self.previous_output_len,
                output_len,
            )
        });
        self.previous_output_len = output_len;
        DecodedStep {
            token_ids,
            logprobs,
            finish_reason,
        }
    }
}

/// Builds SGLang's native sampling dict from vLLM's sampling params.
///
/// Forwarded keys are within sglang v0.5.18 verify() domains, enforced
/// upstream ([`find_rejected_field`]). Keys needing translation:
/// - `top_k` 0 ("all tokens") is omitted; sglang defaults to -1.
/// - `seed` -> `sampling_seed`; the vLLM key name raises TypeError.
/// - `min_tokens` -> `min_new_tokens`; `logit_bias` u32 keys -> strings.
fn to_sglang_sampling(params: &EngineCoreSamplingParams) -> serde_json::Value {
    let mut json = serde_json::json!({
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_new_tokens": params.max_tokens,
        "min_p": params.min_p,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "repetition_penalty": params.repetition_penalty,
    });
    if params.top_k > 0 {
        json["top_k"] = serde_json::json!(params.top_k);
    }
    if params.min_tokens > 0 {
        json["min_new_tokens"] = serde_json::json!(params.min_tokens);
    }
    if let Some(seed) = params.seed {
        json["sampling_seed"] = serde_json::json!(seed);
    }
    if !params.stop_token_ids.is_empty() {
        json["stop_token_ids"] = serde_json::json!(params.stop_token_ids);
    }
    if let Some(logit_bias) = &params.logit_bias {
        json["logit_bias"] = serde_json::Value::Object(
            logit_bias
                .iter()
                .map(|(token_id, bias)| (token_id.to_string(), serde_json::json!(bias)))
                .collect(),
        );
    }
    // `thinking_token_budget` is intentionally unmapped: SGLang's
    // `max_thinking_tokens` (strict-thinking only) is not equivalent.
    json
}

/// sglang v0.5.18 verify() domain bounds.
const TOP_P_MAX: f32 = 1.0; // top_p in (0, 1]
const MIN_P_MAX: f32 = 1.0; // min_p in [0, 1]
const PENALTY_MAX: f32 = 2.0; // freq/presence [-2, 2]; repetition (0, 2]

/// Name of the first field making a request invalid for SGLang: unsupported,
/// outside a verify() domain, or a logprobs mode it cannot honor over the
/// stream. Guarded so the caller gets a clean 400 instead of SGLang failing
/// mid-request.
fn find_rejected_field(params: &EngineCoreSamplingParams) -> Option<&'static str> {
    if params.allowed_token_ids.is_some() {
        return Some("allowed_token_ids");
    }
    if params.bad_words_token_ids.is_some() {
        return Some("bad_words_token_ids");
    }
    if params.repetition_detection.is_some() {
        return Some("repetition_detection");
    }
    if params.structured_outputs.is_some() {
        return Some("structured_outputs");
    }
    if params.skip_reading_prefix_cache.is_some() {
        return Some("skip_reading_prefix_cache");
    }
    // NaN fails these comparisons and is rejected too.
    if params.temperature < 0.0 || !params.temperature.is_finite() {
        return Some("temperature");
    }
    if !(params.top_p > 0.0 && params.top_p <= TOP_P_MAX) {
        return Some("top_p");
    }
    if !(0.0..=MIN_P_MAX).contains(&params.min_p) {
        return Some("min_p");
    }
    if !(-PENALTY_MAX..=PENALTY_MAX).contains(&params.frequency_penalty) {
        return Some("frequency_penalty");
    }
    if !(-PENALTY_MAX..=PENALTY_MAX).contains(&params.presence_penalty) {
        return Some("presence_penalty");
    }
    if !(params.repetition_penalty > 0.0 && params.repetition_penalty <= PENALTY_MAX) {
        return Some("repetition_penalty");
    }
    // Logprobs modes SGLang cannot honor over the stream.
    if params.prompt_logprobs.is_some() {
        return Some("prompt_logprobs");
    }
    if params.logprob_token_ids.is_some() {
        return Some("logprob_token_ids");
    }
    if params.logprobs.is_some_and(|n| n < 0) {
        return Some("logprobs");
    }
    None
}

/// SglangRequest's own top-level fields. An extension key matching one of
/// these would be flattened over the mapped field, silently replacing it, so
/// such keys are rejected rather than let passthrough corrupt the body.
const RESERVED_EXTENSION_KEYS: [&str; 5] = [
    "input_ids",
    "sampling_params",
    "stream",
    "return_logprob",
    "top_logprobs_num",
];

/// Name of the first extension key that collides with a mapped field.
fn find_reserved_extension_key(
    extensions: &BTreeMap<String, serde_json::Value>,
) -> Option<&'static str> {
    RESERVED_EXTENSION_KEYS
        .iter()
        .copied()
        .find(|key| extensions.contains_key(*key))
}

/// SGLang request-level logprob knobs for a vLLM sampling request.
///
/// vLLM's `logprobs` counts the chosen token plus its alternatives; SGLang
/// splits that into `return_logprob` (on/off) and `top_logprobs_num`
/// (alternatives). `None` and `0` both disable logprobs; negatives are
/// rejected upstream by [`find_rejected_field`].
fn to_sglang_logprobs(params: &EngineCoreSamplingParams) -> (Option<bool>, Option<u32>) {
    match params.logprobs {
        None | Some(0) => (None, None),
        Some(n) => (Some(true), Some((n - 1) as u32)),
    }
}
