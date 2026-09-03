// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! SGLang sampling-parameter conversion.
//!
//! Translates vLLM's [`EngineCoreSamplingParams`] into SGLang's native
//! `/generate` sampling dict. Only the shared subset is mapped here:
//! fields SGLang cannot express are rejected upstream at `generate()`
//! ([`find_out_of_contract_field`]) and never reach this conversion;
//! mappable-but-unmapped (`logit_bias`, `min_p`), approximate
//! (`thinking_token_budget`), and derived (`eos_token_id`,
//! `all_stop_token_ids`) fields are intentionally not forwarded
//! (dispositions in ADR-0002).

use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

/// Builds SGLang's native sampling dict from vLLM's sampling params.
///
/// Value domains follow sglang v0.5.18 `SamplingParams` (post_init + verify()):
/// the shared defaults are legal there (temperature >= 0 with [0, 1e-6) forcing
/// greedy, top_p in (0, 1], penalties in [-2, 2]), so those keys are forwarded
/// unconditionally. Only `top_k` and `seed` need translation:
/// - vLLM's `top_k: u32` uses 0 as the "all tokens" sentinel; sglang maps -1 to
///   the whole vocabulary in post_init and verify() rejects values < 1
///   (including 0), so the sentinel is omitted (sglang applies its default).
/// - `seed` is sent as `sampling_seed`: the vLLM key name would raise TypeError
///   inside `SamplingParams`.
pub fn to_sglang_sampling(params: &EngineCoreSamplingParams) -> serde_json::Value {
    let mut json = serde_json::json!({
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_new_tokens": params.max_tokens,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
    });
    if params.top_k > 0 {
        json["top_k"] = serde_json::json!(params.top_k);
    }
    if let Some(seed) = params.seed {
        json["sampling_seed"] = serde_json::json!(seed);
    }
    if !params.stop_token_ids.is_empty() {
        json["stop_token_ids"] = serde_json::json!(params.stop_token_ids);
    }
    json
}

/// Returns the name of the first vLLM-only sampling field that SGLang cannot
/// express, if any is set. Forwarding such a request silently would violate the
/// caller's explicit intent, so the adapter rejects it (HTTP 400).
pub fn find_out_of_contract_field(params: &EngineCoreSamplingParams) -> Option<&'static str> {
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
    None
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
        // Fields outside the shared subset are dropped here, not rejected:
        // `logit_bias` is an S3 key-adaptation field (pending) and
        // `thinking_token_budget` only has an approximate equivalent. Hard
        // out-of-contract fields are rejected upstream (see find_out_of_contract_field).
        let params = EngineCoreSamplingParams {
            thinking_token_budget: Some(128),
            logit_bias: Some(std::collections::HashMap::from([(1234, 1.0_f32)])),
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert!(json.get("thinking_token_budget").is_none());
        assert!(json.get("logit_bias").is_none());
    }

    #[test]
    fn find_out_of_contract_field_detects_rejected_fields() {
        let rejected = EngineCoreSamplingParams {
            allowed_token_ids: Some(vec![5, 6]),
            ..Default::default()
        };
        assert_eq!(
            find_out_of_contract_field(&rejected),
            Some("allowed_token_ids")
        );
        let rejected = EngineCoreSamplingParams {
            bad_words_token_ids: Some(vec![vec![1]]),
            ..Default::default()
        };
        assert_eq!(
            find_out_of_contract_field(&rejected),
            Some("bad_words_token_ids")
        );
        let rejected = EngineCoreSamplingParams {
            skip_reading_prefix_cache: Some(true),
            ..Default::default()
        };
        assert_eq!(
            find_out_of_contract_field(&rejected),
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
        assert_eq!(
            find_out_of_contract_field(&rejected),
            Some("repetition_detection")
        );
        let rejected = EngineCoreSamplingParams {
            structured_outputs: Some(
                vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams::json(
                    serde_json::json!({}),
                ),
            ),
            ..Default::default()
        };
        assert_eq!(
            find_out_of_contract_field(&rejected),
            Some("structured_outputs")
        );
        assert_eq!(
            find_out_of_contract_field(&EngineCoreSamplingParams::default()),
            None
        );
    }
}
