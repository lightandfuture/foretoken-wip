// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! SGLang adapter backed by the engine's loopback HTTP `/generate` endpoint.

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::Deserialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::{Engine, EngineCapabilities, EngineError, EngineTelemetry, TokenStream};
use foretoken_model_protocol::{
    FinishReason, GenerateInput, SamplingParams, TokenEvent, TokenOutput,
};

/// SGLang adapter failures, translated into the engine-neutral [`EngineError`].
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum SglangError {
    #[error("request is invalid")]
    InvalidRequest,
    #[error("sglang is unavailable")]
    Unavailable,
    #[error("sglang protocol failed")]
    Protocol,
    #[error("sglang request failed")]
    RequestFailed,
}

impl From<SglangError> for EngineError {
    fn from(error: SglangError) -> Self {
        match error {
            SglangError::InvalidRequest => EngineError::InvalidRequest,
            SglangError::Unavailable => EngineError::Unavailable,
            SglangError::Protocol => EngineError::Protocol,
            SglangError::RequestFailed => EngineError::RequestFailed,
        }
    }
}

/// Request body for SGLang's native `/generate`.
#[derive(serde::Serialize)]
struct GenerateRequest {
    input_ids: Vec<u32>,
    sampling_params: serde_json::Value,
    stream: bool,
}

/// One streamed token chunk from SGLang.
#[derive(Debug, Deserialize)]
struct GenerateChunk {
    output_ids: Vec<u32>,
    #[serde(default)]
    meta_info: Option<ChunkMeta>,
}

#[derive(Debug, Deserialize)]
struct ChunkMeta {
    /// SGLang reports finish reasons as objects (`{"type":"stop",...}`,
    /// `{"type":"length","length":N}`, ...), so keep the raw value and
    /// extract the reason kind from its `type` field.
    #[serde(default)]
    finish_reason: Option<serde_json::Value>,
}

/// Maps an SGLang finish-reason object into the neutral [`FinishReason`].
fn parse_sglang_finish_reason(reason: &serde_json::Value) -> FinishReason {
    match reason.get("type").and_then(|kind| kind.as_str()) {
        Some("stop") => FinishReason::Stop(None),
        Some("length") => FinishReason::Length,
        Some("abort") => FinishReason::Abort,
        Some("repetition") => FinishReason::Repetition(None),
        _ => FinishReason::Error,
    }
}

/// Parses one line of SGLang's streaming `/generate` response (Server-Sent
/// Events) into a chunk.
///
/// Each payload line looks like `data: {json}`. Lines that carry no payload
/// (blank lines, comment/heartbeat lines, and the `[DONE]` terminator) yield
/// `Ok(None)`; a malformed `data:` payload yields `Err(())`.
fn parse_sse_chunk(line: &[u8]) -> Result<Option<GenerateChunk>, ()> {
    let line = std::str::from_utf8(line).map_err(|_| ())?.trim();
    if line.is_empty() || line.starts_with(':') || line == "[DONE]" {
        return Ok(None);
    }
    let line = line.strip_prefix("data:").unwrap_or(line).trim();
    serde_json::from_str(line).map(Some).map_err(|_| ())
}

/// HTTP-backed SGLang engine.
pub struct SglangBackend {
    client: reqwest::Client,
    endpoint: String,
    running_requests: Arc<AtomicU64>,
}

impl SglangBackend {
    pub fn new(endpoint: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            running_requests: Arc::new(AtomicU64::new(0)),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.endpoint)
    }

    /// Builds the engine-neutral token stream from SGLang's streaming
    /// `/generate` response bytes.
    ///
    /// The first output carries the request's prompt token ids: the frontend's
    /// streaming decoder requires them on the first output to initialize
    /// incremental decoding.
    fn token_stream(
        request_id: String,
        prompt_token_ids: Vec<u32>,
        body: impl Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send + 'static,
    ) -> TokenStream {
        let stream = async_stream::stream! {
            let mut body = body;
            let mut pending: Vec<u8> = Vec::new();
            let mut first_output = true;
            // SGLang streams cumulative `output_ids` (each chunk repeats the
            // whole prefix), so track the previous length and emit only the
            // increment: the frontend's decoder expects per-step deltas.
            let mut previous_output_len = 0_usize;
            while let Some(chunk) = body.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(_) => {
                        yield Err(EngineError::from(SglangError::Protocol));
                        return;
                    }
                };
                pending.extend_from_slice(&chunk);
                while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
                    let line: Vec<u8> = pending.drain(..=newline).collect();
                    let chunk = match parse_sse_chunk(&line[..line.len() - 1]) {
                        Ok(Some(chunk)) => chunk,
                        Ok(None) => continue,
                        Err(()) => {
                            yield Err(EngineError::from(SglangError::Protocol));
                            return;
                        }
                    };
                    let output_len = chunk.output_ids.len();
                    let incremental = chunk
                        .output_ids
                        .get(previous_output_len..)
                        .unwrap_or(&[])
                        .to_vec();
                    previous_output_len = output_len;
                    let finish_reason = chunk
                        .meta_info
                        .and_then(|meta| meta.finish_reason)
                        .map(|reason| parse_sglang_finish_reason(&reason))
                        .or_else(|| incremental.is_empty().then_some(FinishReason::Length));
                    yield Ok(TokenEvent::Token(Box::new(TokenOutput {
                        request_id: request_id.clone(),
                        prompt_token_ids: if first_output {
                            Some(prompt_token_ids.clone())
                        } else {
                            None
                        },
                        prompt_logprobs: None,
                        token_ids: incremental,
                        logprobs: None,
                        cached_token_count: 0,
                        finish_reason,
                        kv_transfer_params: None,
                        ec_transfer_params: None,
                    })));
                    first_output = false;
                }
            }
            if !pending.is_empty() {
                yield Err(EngineError::from(SglangError::Protocol));
            }
        };
        Box::pin(stream)
    }

    /// Builds SGLang's native sampling params from the neutral typed fields
    /// only. Engine-specific keys carried by [`SamplingParams::extra_args`]
    /// are never forwarded: the current frontend fills that map with vLLM
    /// field names, and SGLang rejects them (see ADR-0001). SGLang's own
    /// native extension channel (`custom_params`) will be wired up when the
    /// neutral protocol stops being vLLM-centric.
    fn sampling_json(params: &SamplingParams) -> serde_json::Value {
        let mut json = serde_json::json!({
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_new_tokens": params.max_tokens,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
        });
        if let Some(seed) = params.seed {
            json["seed"] = serde_json::json!(seed);
        }
        if !params.stop_token_ids.is_empty() {
            json["stop_token_ids"] = serde_json::json!(params.stop_token_ids);
        }
        json
    }
}

#[async_trait]
impl Engine for SglangBackend {
    async fn generate(&self, request: GenerateInput) -> Result<TokenStream, EngineError> {
        let request_id = request.request_id.clone();
        let prompt_token_ids = request.prompt_token_ids.clone();
        let sampling = Self::sampling_json(&request.sampling_params);
        let body = GenerateRequest {
            input_ids: prompt_token_ids.clone(),
            sampling_params: sampling,
            stream: true,
        };

        let response = self
            .client
            .post(self.url("/generate"))
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                if error.is_connect() || error.is_timeout() {
                    SglangError::Unavailable
                } else {
                    SglangError::RequestFailed
                }
            })?;
        if !response.status().is_success() {
            return Err(if response.status().is_server_error() {
                SglangError::Unavailable
            } else {
                SglangError::InvalidRequest
            }
            .into());
        }

        let running_requests = self.running_requests.clone();
        running_requests.fetch_add(1, Ordering::AcqRel);
        let guard = RunningGuard { running_requests };
        let stream = Self::token_stream(request_id, prompt_token_ids, response.bytes_stream());
        let stream = stream.scan(guard, |_guard, event| async move { Some(event) });

        Ok(Box::pin(stream))
    }

    async fn abort(&self, _request_ids: &[String]) -> Result<(), EngineError> {
        // SGLang does not expose a stable per-request abort endpoint for the
        // native `/generate` path; report success to keep the core contract.
        Ok(())
    }

    fn telemetry(&self) -> EngineTelemetry {
        EngineTelemetry {
            running_requests: self.running_requests.load(Ordering::Acquire),
            ..Default::default()
        }
    }

    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities::default()
    }

    async fn cleanup(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

/// Decrements the running-request counter when a stream completes or is dropped.
struct RunningGuard {
    running_requests: Arc<AtomicU64>,
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.running_requests.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn sampling_json_drops_vllm_specific_extra_args() {
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
        let json = SglangBackend::sampling_json(&params);
        assert!(json.get("all_stop_token_ids").is_none());
        assert!(json.get("structured_outputs").is_none());
        assert!(json.get("logit_bias").is_none());
    }

    #[test]
    fn sampling_json_keeps_neutral_fields() {
        let params = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            seed: Some(42),
            max_tokens: 128,
            stop_token_ids: vec![151643],
            ..Default::default()
        };
        let json = SglangBackend::sampling_json(&params);
        assert_eq!(json["temperature"].as_f64().unwrap() as f32, 0.7);
        assert_eq!(json["top_p"].as_f64().unwrap() as f32, 0.9);
        assert_eq!(json["top_k"], 40);
        assert_eq!(json["seed"], 42);
        assert_eq!(json["max_new_tokens"], 128);
        assert_eq!(json["stop_token_ids"], serde_json::json!([151643]));
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

    #[tokio::test]
    async fn first_token_output_carries_prompt_token_ids() {
        use futures::stream;
        // SGLang streams cumulative output_ids, so the second chunk repeats
        // the first chunk's prefix.
        let body = stream::iter(vec![
            Ok::<_, reqwest::Error>(Bytes::from_static(b"data: {\"output_ids\":[1]}\n")),
            Ok::<_, reqwest::Error>(Bytes::from_static(
                b"data: {\"output_ids\":[1,2],\"meta_info\":{\"finish_reason\":{\"type\":\"stop\"}}}\n",
            )),
        ]);
        let mut events = SglangBackend::token_stream("r1".into(), vec![100, 200], body);
        let first = events.next().await.expect("first event");
        match first {
            Ok(TokenEvent::Token(token)) => {
                assert_eq!(token.prompt_token_ids.as_deref(), Some(&[100, 200][..]));
                assert_eq!(token.token_ids, vec![1]);
                assert_eq!(token.finish_reason, None);
            }
            _ => panic!("expected first token output"),
        }
        let second = events.next().await.expect("second event");
        match second {
            Ok(TokenEvent::Token(token)) => {
                assert!(token.prompt_token_ids.is_none());
                assert_eq!(token.token_ids, vec![2], "cumulative ids must become deltas");
                assert_eq!(token.finish_reason, Some(FinishReason::Stop(None)));
            }
            _ => panic!("expected second token output"),
        }
        assert!(events.next().await.is_none());
    }

    #[test]
    fn parses_object_finish_reason() {
        assert_eq!(
            parse_sglang_finish_reason(&serde_json::json!({"type":"stop","stop_reason":"<|im_end|>"})),
            FinishReason::Stop(None)
        );
        assert_eq!(
            parse_sglang_finish_reason(&serde_json::json!({"type":"length","length":5})),
            FinishReason::Length
        );
        assert_eq!(
            parse_sglang_finish_reason(&serde_json::json!({"type":"abort","abort_reason":"killed"})),
            FinishReason::Abort
        );
        assert_eq!(parse_sglang_finish_reason(&serde_json::json!({})), FinishReason::Error);
    }
}
