// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! SGLang adapter contract tests using a mock loopback HTTP server.

#![cfg(feature = "backend-sglang")]

use axum::{Json, Router, routing::post};
use foretoken_model_server::engine::Engine;
use foretoken_model_server::engine::sglang::{SglangBackend, SglangLaunchPlan};
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use vllm_llm::{FinishReason, GenerateRequest};

fn generate_request() -> GenerateRequest {
    GenerateRequest {
        request_id: "req".into(),
        prompt_token_ids: vec![1, 2],
        sampling_params: Default::default(),
        mm_features: None,
        arrival_time: None,
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        session_id: None,
        reasoning_parser_kwargs: None,
        lora_request: None,
    }
}

/// Builds a mock SGLang server that streams the given NDJSON response body.
async fn spawn_mock_server(response_body: String) -> (String, Arc<Mutex<Value>>) {
    let last_request: Arc<Mutex<Value>> = Arc::new(Mutex::new(Value::Null));

    let seen = last_request.clone();
    let app = Router::new().route(
        "/generate",
        post(move |Json(body): Json<Value>| {
            let seen = seen.clone();
            let response_body = response_body.clone();
            async move {
                *seen.lock().unwrap() = body;
                (
                    axum::http::StatusCode::OK,
                    [("content-type", "application/json")],
                    response_body,
                )
            }
        }),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), last_request)
}

/// Real SGLang streams cumulative `output_ids` (each chunk repeats the
/// prefix) and reports object-shaped finish reasons (`{"type": "stop", ...}`).
fn real_sglang_body() -> String {
    format!(
        "{}\n{}\n",
        json!({"output_ids": [42]}),
        json!({"output_ids": [42, 43], "meta_info": {"finish_reason": {"type": "stop"}}})
    )
}

#[tokio::test]
async fn generate_streams_tokens_and_terminal() {
    let (endpoint, seen) = spawn_mock_server(real_sglang_body()).await;
    let backend = SglangBackend::new(endpoint);

    let mut stream = backend.generate(generate_request()).await.unwrap();
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }

    assert_eq!(events.len(), 2);
    // The first output carries the prompt ids for the frontend decoder.
    assert_eq!(
        events[0].prompt_token_ids().map(|ids| ids.as_ref()),
        Some(&[1, 2][..])
    );
    assert_eq!(events[0].token_ids, vec![42]);
    assert_eq!(events[0].finish_reason, None);
    assert_eq!(
        events[1].token_ids,
        vec![43],
        "cumulative ids must become deltas"
    );
    assert_eq!(events[1].finish_reason, Some(FinishReason::Stop(None)));

    // The mock saw the tokenized request.
    let body = seen.lock().unwrap();
    assert_eq!(body["input_ids"], json!([1, 2]));
    assert_eq!(body["stream"], json!(true));
}

#[tokio::test]
async fn generate_maps_object_length_finish_reason() {
    // A single chunk carrying an object-shaped `length` finish reason, as the
    // real server reports when max_new_tokens is exhausted.
    let body = format!(
        "{}\n",
        json!({"output_ids": [42], "meta_info": {"finish_reason": {"type": "length", "length": 1}}})
    );
    let (endpoint, _seen) = spawn_mock_server(body).await;
    let backend = SglangBackend::new(endpoint);

    let mut stream = backend.generate(generate_request()).await.unwrap();
    let output = stream.next().await.unwrap().unwrap();
    assert_eq!(output.token_ids, vec![42]);
    assert_eq!(output.finish_reason, Some(FinishReason::Length));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn generate_reports_backend_error_on_non_success() {
    let app = Router::new().route(
        "/generate",
        post(|| async { axum::http::StatusCode::SERVICE_UNAVAILABLE }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let backend = SglangBackend::new(format!("http://{addr}"));
    let result = backend.generate(generate_request()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn default_capabilities_and_cleanup() {
    let backend = SglangBackend::new("http://127.0.0.1:1".to_owned());
    let capabilities = backend.capabilities();
    assert!(!capabilities.kv_event_sources);
    assert!(!capabilities.supports_pd);
    assert!(!capabilities.supports_ec);
    assert!(backend.cleanup().await.is_ok());
}

#[test]
fn launch_plan_parses_and_defaults() {
    let plan = SglangLaunchPlan::parse(
        r#"{"version":1,"model":"Qwen/Qwen3-0.6B","port":30000,"startupSeconds":120,"drainSeconds":30}"#,
    )
    .unwrap();
    assert_eq!(plan.tp, 1);
    assert_eq!(plan.dp, 1);
    assert!(
        plan.render_args()
            .unwrap()
            .contains(&"--model-path=Qwen/Qwen3-0.6B".to_string())
    );
}
