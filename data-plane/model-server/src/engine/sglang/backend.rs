// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! SGLang adapter backed by the engine's loopback HTTP `/generate` endpoint.

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::{Engine, EngineCapabilities, EngineError, EngineTelemetry, TokenStream};
use vllm_llm::{FinishReason, GenerateOutput, GeneratePromptInfo};

use super::conversion::{SglangRequest, SglangResponseDecoder, parse_sse_chunk};

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
            let mut decoder = SglangResponseDecoder::default();
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
                    let step = decoder.decode(chunk);
                    if step.finish_reason == Some(FinishReason::Error) {
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
                        token_ids: step.token_ids,
                        logprobs: step.logprobs,
                        finish_reason: step.finish_reason,
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
        let body = SglangRequest::try_from(&request).map_err(|field| {
            tracing::warn!(field, "rejecting field SGLang cannot honor");
            EngineError::InvalidRequest
        })?;
        let request_id = request.request_id.clone();
        let prompt_token_ids = request.prompt_token_ids.clone();

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
}
