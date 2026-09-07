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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_shared_subset() {
        let params = EngineCoreSamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            seed: Some(42),
            max_tokens: 128,
            stop_token_ids: vec![151643],
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert_eq!(json["temperature"].as_f64().unwrap() as f32, 0.7);
        assert_eq!(json["top_p"].as_f64().unwrap() as f32, 0.9);
        assert_eq!(json["top_k"], 40);
        assert!(
            json.get("seed").is_none(),
            "vLLM-style `seed` key must not reach SGLang"
        );
        assert_eq!(json["sampling_seed"], 42);
        assert_eq!(json["max_new_tokens"], 128);
        assert_eq!(json["stop_token_ids"], serde_json::json!([151643]));
    }

    #[test]
    fn omits_top_k_zero_sentinel() {
        // Guards the top_k=0 sentinel translation (see to_sglang_sampling).
        let params = EngineCoreSamplingParams {
            top_k: 0, // also the default value
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert!(
            json.get("top_k").is_none(),
            "top_k=0 sentinel must not reach SGLang: {json}"
        );
    }

    #[test]
    fn forwards_explicit_top_k() {
        let params = EngineCoreSamplingParams {
            top_k: 5,
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert_eq!(json["top_k"], 5);
    }

    #[test]
    fn uses_sglang_seed_key_name() {
        // Guards the seed -> sampling_seed key translation (see to_sglang_sampling).
        let params = EngineCoreSamplingParams {
            seed: Some(42),
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert!(json.get("seed").is_none());
        assert_eq!(json["sampling_seed"], 42);
    }

    #[test]
    fn drops_unmapped_vllm_fields() {
        // Unmapped, not rejected: `logprobs` is request-level in SGLang;
        // `thinking_token_budget` has no exact equivalent.
        let params = EngineCoreSamplingParams {
            thinking_token_budget: Some(128),
            logprobs: Some(5),
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert!(json.get("thinking_token_budget").is_none());
        assert!(json.get("logprobs").is_none());
    }

    #[test]
    fn maps_contract_mappable_fields() {
        let params = EngineCoreSamplingParams {
            min_tokens: 16,
            min_p: 0.1,
            repetition_penalty: 1.2,
            logit_bias: Some(std::collections::HashMap::from([(1234, 1.5_f32)])),
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert_eq!(json["min_new_tokens"], 16);
        assert!(json.get("min_tokens").is_none(), "vLLM key must not leak");
        assert_eq!(json["min_p"].as_f64().unwrap() as f32, 0.1);
        assert_eq!(json["repetition_penalty"].as_f64().unwrap() as f32, 1.2);
        assert_eq!(json["logit_bias"]["1234"], serde_json::json!(1.5));
        assert!(
            to_sglang_sampling(&EngineCoreSamplingParams::default())
                .get("min_new_tokens")
                .is_none(),
            "default min_tokens must be omitted"
        );
    }

    #[test]
    fn find_rejected_field_detects_out_of_range_values() {
        for (params, expected) in [
            (
                EngineCoreSamplingParams {
                    temperature: -0.5,
                    ..Default::default()
                },
                "temperature",
            ),
            (
                EngineCoreSamplingParams {
                    top_p: 0.0,
                    ..Default::default()
                },
                "top_p",
            ),
            (
                EngineCoreSamplingParams {
                    top_p: 1.5,
                    ..Default::default()
                },
                "top_p",
            ),
            (
                EngineCoreSamplingParams {
                    min_p: 1.5,
                    ..Default::default()
                },
                "min_p",
            ),
            (
                EngineCoreSamplingParams {
                    frequency_penalty: 3.0,
                    ..Default::default()
                },
                "frequency_penalty",
            ),
            (
                EngineCoreSamplingParams {
                    presence_penalty: -3.0,
                    ..Default::default()
                },
                "presence_penalty",
            ),
            (
                EngineCoreSamplingParams {
                    repetition_penalty: 0.0,
                    ..Default::default()
                },
                "repetition_penalty",
            ),
            (
                EngineCoreSamplingParams {
                    repetition_penalty: 2.5,
                    ..Default::default()
                },
                "repetition_penalty",
            ),
            // NaN must not slip past the guards into sglang verify().
            (
                EngineCoreSamplingParams {
                    min_p: f32::NAN,
                    ..Default::default()
                },
                "min_p",
            ),
            (
                EngineCoreSamplingParams {
                    repetition_penalty: f32::NAN,
                    ..Default::default()
                },
                "repetition_penalty",
            ),
        ] {
            assert_eq!(find_rejected_field(&params), Some(expected));
        }
        // In-domain edges are accepted.
        for params in [
            EngineCoreSamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            EngineCoreSamplingParams {
                top_p: 1.0,
                ..Default::default()
            },
            EngineCoreSamplingParams {
                min_p: 1.0,
                ..Default::default()
            },
            EngineCoreSamplingParams {
                frequency_penalty: -2.0,
                ..Default::default()
            },
            EngineCoreSamplingParams {
                presence_penalty: 2.0,
                ..Default::default()
            },
            EngineCoreSamplingParams {
                repetition_penalty: 2.0,
                ..Default::default()
            },
        ] {
            assert_eq!(find_rejected_field(&params), None);
        }
        assert_eq!(
            find_rejected_field(&EngineCoreSamplingParams::default()),
            None
        );
    }

    #[test]
    fn find_rejected_field_rejects_unhonorable_logprobs_modes() {
        let params = |f: &dyn Fn(&mut EngineCoreSamplingParams)| {
            let mut p = EngineCoreSamplingParams::default();
            f(&mut p);
            p
        };
        assert_eq!(
            find_rejected_field(&params(&|p| p.prompt_logprobs = Some(5))),
            Some("prompt_logprobs")
        );
        assert_eq!(
            find_rejected_field(&params(&|p| p.logprobs = Some(-1))),
            Some("logprobs")
        );
        assert_eq!(
            find_rejected_field(&params(&|p| p.logprob_token_ids = Some(vec![7]))),
            Some("logprob_token_ids")
        );
        assert_eq!(
            find_rejected_field(&params(&|p| {
                p.logprobs = Some(3);
                p.logprob_token_ids = Some(vec![7]);
            })),
            Some("logprob_token_ids")
        );
    }

    #[test]
    fn to_sglang_logprobs_maps_count_to_knobs() {
        assert_eq!(
            to_sglang_logprobs(&EngineCoreSamplingParams::default()),
            (None, None)
        );
        assert_eq!(
            to_sglang_logprobs(&EngineCoreSamplingParams {
                logprobs: Some(0),
                ..Default::default()
            }),
            (None, None)
        );
        assert_eq!(
            to_sglang_logprobs(&EngineCoreSamplingParams {
                logprobs: Some(1),
                ..Default::default()
            }),
            (Some(true), Some(0))
        );
        assert_eq!(
            to_sglang_logprobs(&EngineCoreSamplingParams {
                logprobs: Some(5),
                ..Default::default()
            }),
            (Some(true), Some(4))
        );
    }

    #[test]
    fn find_rejected_field_detects_unsupported_fields() {
        let rejected = EngineCoreSamplingParams {
            allowed_token_ids: Some(vec![5, 6]),
            ..Default::default()
        };
        assert_eq!(find_rejected_field(&rejected), Some("allowed_token_ids"));
        let rejected = EngineCoreSamplingParams {
            bad_words_token_ids: Some(vec![vec![1]]),
            ..Default::default()
        };
        assert_eq!(find_rejected_field(&rejected), Some("bad_words_token_ids"));
        let rejected = EngineCoreSamplingParams {
            skip_reading_prefix_cache: Some(true),
            ..Default::default()
        };
        assert_eq!(
            find_rejected_field(&rejected),
            Some("skip_reading_prefix_cache")
        );
        let rejected = EngineCoreSamplingParams {
            repetition_detection: Some(
                vllm_engine_core_client::protocol::sampling::RepetitionDetectionParams {
                    max_pattern_size: 5,
                    min_pattern_size: 0,
                    min_count: 2,
                },
            ),
            ..Default::default()
        };
        assert_eq!(find_rejected_field(&rejected), Some("repetition_detection"));
        let rejected = EngineCoreSamplingParams {
            structured_outputs: Some(
                vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams::json(
                    serde_json::json!({}),
                ),
            ),
            ..Default::default()
        };
        assert_eq!(find_rejected_field(&rejected), Some("structured_outputs"));
        assert_eq!(
            find_rejected_field(&EngineCoreSamplingParams::default()),
            None
        );
    }

    #[test]
    fn sglang_request_maps_vllm_fields() {
        let request = GenerateRequest {
            prompt_token_ids: vec![7, 8],
            sampling_params: EngineCoreSamplingParams {
                temperature: 0.7,
                logprobs: Some(3),
                ..Default::default()
            },
            ..Default::default()
        };
        let body = SglangRequest::try_from(&request).expect("valid request");
        assert_eq!(body.input_ids, vec![7, 8]);
        assert!(body.stream);
        assert_eq!(body.return_logprob, Some(true));
        assert_eq!(body.top_logprobs_num, Some(2));
        assert_eq!(
            body.sampling_params["temperature"].as_f64().unwrap() as f32,
            0.7
        );
    }

    #[test]
    fn sglang_request_merges_extensions_into_the_body() {
        let request = GenerateRequest {
            prompt_token_ids: vec![7],
            extensions: std::collections::BTreeMap::from([
                (
                    "session_params".to_string(),
                    serde_json::json!({"id": "s1"}),
                ),
                ("custom_params".to_string(), serde_json::json!({"k": 1})),
            ]),
            ..Default::default()
        };
        let body = SglangRequest::try_from(&request).expect("valid request");
        let json = serde_json::to_value(&body).expect("serializes");
        assert_eq!(json["session_params"], serde_json::json!({"id": "s1"}));
        assert_eq!(json["custom_params"], serde_json::json!({"k": 1}));
        assert_eq!(json["input_ids"], serde_json::json!([7]));
    }

    #[test]
    fn sglang_request_rejects_reserved_extension_keys() {
        let request = GenerateRequest {
            prompt_token_ids: vec![7],
            extensions: std::collections::BTreeMap::from([(
                "input_ids".to_string(),
                serde_json::json!([1, 2]),
            )]),
            ..Default::default()
        };
        assert!(matches!(
            SglangRequest::try_from(&request),
            Err("input_ids")
        ));
    }

    #[test]
    fn sglang_request_rejects_unhonorable_fields() {
        let request = GenerateRequest {
            sampling_params: EngineCoreSamplingParams {
                prompt_logprobs: Some(5),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            SglangRequest::try_from(&request),
            Err("prompt_logprobs")
        ));
    }

    #[test]
    fn decoder_windows_cumulative_chunks() {
        let mut decoder = SglangResponseDecoder::default();
        let first = decoder.decode(SglangChunk {
            output_ids: vec![11],
            meta_info: None,
        });
        assert_eq!(first.token_ids, vec![11]);
        assert!(first.logprobs.is_none());
        assert_eq!(first.finish_reason, None);

        let second = decoder.decode(SglangChunk {
            output_ids: vec![11, 12],
            meta_info: Some(SglangChunkMeta {
                finish_reason: Some(serde_json::json!({"type": "stop"})),
                output_token_logprobs: vec![(-0.5, 11, None), (-0.2, 12, None)],
                output_top_logprobs: vec![vec![], vec![]],
            }),
        });
        assert_eq!(second.token_ids, vec![12]);
        assert_eq!(second.finish_reason, Some(FinishReason::Stop(None)));
        let logprobs = second.logprobs.expect("delta carries its logprobs");
        assert_eq!(logprobs.positions.len(), 1);
        assert_eq!(logprobs.positions[0].entries[0].token_id, 12);
    }

    #[test]
    fn parse_sse_chunk_strips_data_prefix() {
        let chunk = parse_sse_chunk(
            br#"data: {"text":", I","output_ids":[11,358],"meta_info":{"finish_reason":null}}"#,
        )
        .expect("valid SSE payload")
        .expect("some chunk");
        assert_eq!(chunk.output_ids, vec![11, 358]);
        assert!(chunk.meta_info.is_some());
    }

    #[test]
    fn parse_sse_chunk_skips_framing_lines() {
        assert!(parse_sse_chunk(b"").unwrap().is_none());
        assert!(parse_sse_chunk(b" ").unwrap().is_none());
        assert!(parse_sse_chunk(b": ping").unwrap().is_none());
        assert!(parse_sse_chunk(b"[DONE]").unwrap().is_none());
    }

    #[test]
    fn parse_sse_chunk_rejects_malformed_payload() {
        assert!(parse_sse_chunk(b"data: {not-json").is_err());
    }

    #[test]
    fn parse_sse_chunk_reads_logprobs_meta() {
        let chunk = parse_sse_chunk(
            br#"data: {"output_ids":[11,12],"meta_info":{"output_token_logprobs":[[-0.5,11,null],[-0.2,12,"a"]],"output_top_logprobs":[[[-0.5,11,null],[-0.9,99,null]],[[-0.2,12,null]]],"finish_reason":null}}"#,
        )
        .expect("valid SSE payload")
        .expect("some chunk");
        let meta = chunk.meta_info.expect("meta present");
        assert_eq!(meta.output_token_logprobs.len(), 2);
        assert_eq!(meta.output_token_logprobs[1], (-0.2, 12, Some("a".into())));
        assert_eq!(meta.output_top_logprobs[0].len(), 2);
        assert_eq!(meta.output_top_logprobs[1].len(), 1);
    }

    #[test]
    fn chunk_logprobs_windows_to_the_token_delta() {
        let best = vec![(-0.5, 11, None), (-0.2, 12, None), (-0.1, 13, None)];
        let top = vec![
            vec![(-0.5, 11, None)],
            // Repeats the chosen token (12); it must be deduped.
            vec![(-0.2, 12, None), (-0.4, 99, None)],
            vec![],
        ];
        // The stream had already emitted position 0; the window covers 1..3.
        let logprobs = chunk_logprobs(&best, &top, 1, 3).expect("window");
        assert_eq!(logprobs.positions.len(), 2);
        let second = &logprobs.positions[0];
        assert_eq!(second.entries.len(), 2);
        assert_eq!(second.entries[0].token_id, 12);
        assert_eq!(second.entries[0].rank, 1);
        assert_eq!(second.entries[1].token_id, 99);
        assert_eq!(second.entries[1].rank, 2);
        assert_eq!(logprobs.positions[1].entries.len(), 1);
    }

    #[test]
    fn chunk_logprobs_none_when_not_requested() {
        // sglang sends empty arrays when logprobs were not requested.
        assert_eq!(chunk_logprobs(&[], &[], 0, 2), None);
        assert_eq!(chunk_logprobs(&[(-0.5, 11, None)], &[], 1, 2), None);
    }

    #[test]
    fn parses_object_finish_reason() {
        assert_eq!(
            parse_sglang_finish_reason(
                &serde_json::json!({"type":"stop","stop_reason":"<|im_end|>"})
            ),
            FinishReason::Stop(None)
        );
        assert_eq!(
            parse_sglang_finish_reason(&serde_json::json!({"type":"length","length":5})),
            FinishReason::Length
        );
        assert_eq!(
            parse_sglang_finish_reason(
                &serde_json::json!({"type":"abort","abort_reason":"killed"})
            ),
            FinishReason::Abort
        );
        assert_eq!(
            parse_sglang_finish_reason(&serde_json::json!({})),
            FinishReason::Error
        );
    }
}
