// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! SGLang sampling-parameter conversion.
//!
//! The neutral `foretoken_model_protocol` types carry no engine dependency;
//! this module is the sole place where the neutral [`SamplingParams`] are
//! translated into SGLang's native `/generate` sampling dict.
//!
//! [`SamplingParams::extra_args`] is ignored here: the frontend fills it with
//! vLLM field names, which SGLang rejects (ADR-0001).

use foretoken_model_protocol::SamplingParams;

/// Builds SGLang's native sampling dict from the neutral typed fields.
///
/// Value domains follow sglang v0.5.18 `SamplingParams` (post_init +
/// verify()): the neutral defaults are legal there (temperature >= 0 with
/// [0, 1e-6) forcing greedy, top_p in (0, 1], penalties in [-2, 2]), so those
/// keys are forwarded unconditionally. Only `top_k` and `seed` need
/// translation:
/// - neutral `top_k: u32` uses 0 as the vLLM-style "all tokens" sentinel;
///   sglang maps -1 to the whole vocabulary in post_init and verify() rejects
///   values < 1 (including 0), so the sentinel is translated by omitting the
///   key: sglang then applies its whole-vocabulary default.
/// - `seed` is sent as `sampling_seed`: the neutral key name would raise
///   TypeError inside `SamplingParams`.
pub fn to_sglang_sampling(params: &SamplingParams) -> serde_json::Value {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn drops_vllm_specific_extra_args() {
        let mut extra = BTreeMap::new();
        extra.insert(
            "all_stop_token_ids".into(),
            serde_json::json!([151643, 151645]),
        );
        extra.insert("structured_outputs".into(), serde_json::json!({}));
        extra.insert("logit_bias".into(), serde_json::json!({}));
        let params = SamplingParams {
            temperature: 0.5,
            max_tokens: 32,
            stop_token_ids: vec![7],
            extra_args: extra,
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert!(json.get("all_stop_token_ids").is_none());
        assert!(json.get("structured_outputs").is_none());
        assert!(json.get("logit_bias").is_none());
    }

    #[test]
    fn keeps_neutral_fields() {
        let params = SamplingParams {
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
        let params = SamplingParams {
            top_k: 0, // also the SamplingParams::default() value
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
        let params = SamplingParams {
            top_k: 5,
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert_eq!(json["top_k"], 5);
    }

    #[test]
    fn uses_sglang_seed_key_name() {
        // Guards the seed -> sampling_seed key translation (see to_sglang_sampling).
        let params = SamplingParams {
            seed: Some(42),
            ..Default::default()
        };
        let json = to_sglang_sampling(&params);
        assert!(json.get("seed").is_none());
        assert_eq!(json["sampling_seed"], 42);
    }
}
