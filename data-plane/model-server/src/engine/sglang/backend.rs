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
use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs, TokenLogprob};
use vllm_llm::{FinishReason, GenerateOutput, GeneratePromptInfo};

use super::conversion::{find_rejected_field, to_sglang_logprobs, to_sglang_sampling};

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
    /// Request-level logprob knobs; omitted unless `logprobs` was set.
    #[serde(skip_serializing_if = "Option::is_none")]
    return_logprob: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs_num: Option<u32>,
}

/// SGLang logprob triple: `[logprob, token_id, token_text]`. Token text is
/// null unless the request set `return_text_in_logprobs`.
type LogprobTriple = (f32, u32, Option<String>);

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
    /// Cumulative per-generated-token logprobs, aligned with `output_ids`;
    /// empty when the request did not ask for logprobs.
    #[serde(default)]
    output_token_logprobs: Vec<LogprobTriple>,
    /// Cumulative top-k alternatives per generated token.
    #[serde(default)]
    output_top_logprobs: Vec<Vec<LogprobTriple>>,
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
                    let meta = chunk.meta_info.as_ref();
                    let finish_reason = meta
                        .and_then(|meta| meta.finish_reason.as_ref())
                        .map(parse_sglang_finish_reason)
                        .or_else(|| incremental.is_empty().then_some(FinishReason::Length));
                    let logprobs = meta.and_then(|meta| {
                        chunk_logprobs(
                            &meta.output_token_logprobs,
                            &meta.output_top_logprobs,
                            previous_output_len,
                            output_len,
                        )
                    });
                    previous_output_len = output_len;
                    if finish_reason == Some(FinishReason::Error) {
                        yield Err(EngineError::RequestFailed);
                        return;
                    }
                    yield Ok(GenerateOutput {
                        request_id: request_id.clone(),
                        prompt_info: if first_output {
                            Some(GeneratePromptInfo {
                                prompt_token_ids: prompt_token_ids.clone().into(),
                                prompt_logprobs: None,
                            })
                        } else {
                            None
                        },
                        token_ids: incremental,
                        logprobs,
                        finish_reason,
                        cached_token_count: 0,
                        kv_transfer_params: None,
                        ec_transfer_params: None,
                    });
                    first_output = false;
                }
            }
            if !pending.is_empty() {
                yield Err(EngineError::from(SglangError::Protocol));
            }
        };
        Box::pin(stream)
    }
}

#[async_trait]
impl Engine for SglangBackend {
    async fn generate(
        &self,
        request: vllm_llm::GenerateRequest,
    ) -> Result<TokenStream, EngineError> {
        if let Some(field) = find_rejected_field(&request.sampling_params) {
            tracing::warn!(field, "rejecting sampling field SGLang cannot honor");
            return Err(EngineError::InvalidRequest);
        }
        let request_id = request.request_id.clone();
        let prompt_token_ids = request.prompt_token_ids.clone();
        let sampling = to_sglang_sampling(&request.sampling_params);
        let (return_logprob, top_logprobs_num) = to_sglang_logprobs(&request.sampling_params);
        let body = GenerateRequest {
            input_ids: prompt_token_ids.clone(),
            sampling_params: sampling,
            stream: true,
            return_logprob,
            top_logprobs_num,
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

    #[tokio::test]
    async fn stream_attaches_logprobs_to_the_delta_output() {
        use futures::stream;
        // SGLang streams cumulative output_ids and cumulative logprobs, so
        // the second chunk repeats the first chunk's prefix.
        let body = stream::iter(vec![
            Ok::<_, reqwest::Error>(Bytes::from_static(b"data: {\"output_ids\":[11]}\n")),
            Ok::<_, reqwest::Error>(Bytes::from_static(
                b"data: {\"output_ids\":[11,12],\"meta_info\":{\"output_token_logprobs\":[[-0.5,11,null],[-0.2,12,null]],\"finish_reason\":null}}\n",
            )),
        ]);
        let mut events = SglangBackend::token_stream("r1".into(), vec![100], body);
        let first = events.next().await.expect("first event").expect("ok");
        assert_eq!(first.token_ids, vec![11]);
        assert!(first.logprobs.is_none());
        let second = events.next().await.expect("second event").expect("ok");
        assert_eq!(second.token_ids, vec![12]);
        let logprobs = second.logprobs.expect("delta carries its logprobs");
        assert_eq!(logprobs.positions.len(), 1);
        assert_eq!(logprobs.positions[0].entries[0].token_id, 12);
        assert_eq!(logprobs.positions[0].entries[0].logprob, -0.2);
    }

    #[tokio::test]
    async fn stream_leaves_logprobs_none_when_not_requested() {
        use futures::stream;
        let body = stream::iter(vec![
            Ok::<_, reqwest::Error>(Bytes::from_static(
                b"data: {\"output_ids\":[11,12],\"meta_info\":{\"output_token_logprobs\":[],\"finish_reason\":null}}\n",
            )),
        ]);
        let mut events = SglangBackend::token_stream("r1".into(), vec![100], body);
        let first = events.next().await.expect("event").expect("ok");
        assert_eq!(first.token_ids, vec![11, 12]);
        assert!(first.logprobs.is_none());
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
            Ok(token) => {
                assert_eq!(
                    token.prompt_token_ids().map(|ids| ids.as_ref()),
                    Some(&[100, 200][..])
                );
                assert_eq!(token.token_ids, vec![1]);
                assert_eq!(token.finish_reason, None);
            }
            _ => panic!("expected first token output"),
        }
        let second = events.next().await.expect("second event");
        match second {
            Ok(token) => {
                assert!(token.prompt_info.is_none());
                assert_eq!(
                    token.token_ids,
                    vec![2],
                    "cumulative ids must become deltas"
                );
                assert_eq!(token.finish_reason, Some(FinishReason::Stop(None)));
            }
            _ => panic!("expected second token output"),
        }
        assert!(events.next().await.is_none());
    }

    #[tokio::test]
    async fn error_finish_reason_yields_request_failed() {
        use futures::stream;
        // An unrecognized finish-reason type maps to FinishReason::Error and
        // must surface as a request failure, not a successful output.
        let body = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from_static(
            b"data: {\"output_ids\":[1],\"meta_info\":{\"finish_reason\":{\"type\":\"unrecognized\"}}}\n",
        ))]);
        let mut events = SglangBackend::token_stream("r1".into(), vec![100, 200], body);
        match events.next().await.expect("first event") {
            Err(EngineError::RequestFailed) => {}
            other => panic!("expected RequestFailed, got {other:?}"),
        }
        assert!(events.next().await.is_none());
    }

    #[tokio::test]
    async fn rejects_sampling_fields_sglang_cannot_honor() {
        // Rejected before any HTTP; the request never reaches SGLang.
        let backend = SglangBackend::new("http://127.0.0.1:1".into());
        for params in [
            vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams {
                allowed_token_ids: Some(vec![5]),
                ..Default::default()
            },
            vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams {
                bad_words_token_ids: Some(vec![vec![1]]),
                ..Default::default()
            },
            vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams {
                repetition_detection: Some(
                    vllm_engine_core_client::protocol::sampling::RepetitionDetectionParams {
                        max_pattern_size: 5,
                        min_pattern_size: 0,
                        min_count: 2,
                    },
                ),
                ..Default::default()
            },
            vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams {
                structured_outputs: Some(
                    vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams::json(
                        serde_json::json!({}),
                    ),
                ),
                ..Default::default()
            },
            vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams {
                skip_reading_prefix_cache: Some(true),
                ..Default::default()
            },
            vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams {
                min_p: 1.5, // outside SGLang's [0, 1] domain
                ..Default::default()
            },
        ] {
            let request = vllm_llm::GenerateRequest {
                sampling_params: params,
                ..Default::default()
            };
            match backend.generate(request).await {
                Err(EngineError::InvalidRequest) => {}
                Err(other) => panic!("expected InvalidRequest, got {other:?}"),
                Ok(_) => panic!("expected InvalidRequest, got an output stream"),
            }
        }
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
