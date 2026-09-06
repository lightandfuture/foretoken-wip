// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! SGLang sampling-parameter conversion.
//!
//! Maps vLLM's [`EngineCoreSamplingParams`] to SGLang's `/generate` sampling
//! dict. Requests SGLang cannot honor are rejected at `generate()`
//! ([`find_rejected_field`]); `logprobs` maps to request-level knobs
//! ([`to_sglang_logprobs`]); `thinking_token_budget` and engine-derived stop
//! fields are not forwarded.

use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

/// Builds SGLang's native sampling dict from vLLM's sampling params.
///
/// Forwarded keys are within sglang v0.5.18 verify() domains, enforced
/// upstream ([`find_rejected_field`]). Keys needing translation:
/// - `top_k` 0 ("all tokens") is omitted; sglang defaults to -1.
/// - `seed` -> `sampling_seed`; the vLLM key name raises TypeError.
/// - `min_tokens` -> `min_new_tokens`; `logit_bias` u32 keys -> strings.
pub fn to_sglang_sampling(params: &EngineCoreSamplingParams) -> serde_json::Value {
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
pub fn find_rejected_field(params: &EngineCoreSamplingParams) -> Option<&'static str> {
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

/// SGLang request-level logprob knobs for a vLLM sampling request.
///
/// vLLM's `logprobs` counts the chosen token plus its alternatives; SGLang
/// splits that into `return_logprob` (on/off) and `top_logprobs_num`
/// (alternatives). `None` and `0` both disable logprobs; negatives are
/// rejected upstream by [`find_rejected_field`].
pub fn to_sglang_logprobs(params: &EngineCoreSamplingParams) -> (Option<bool>, Option<u32>) {
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
}
