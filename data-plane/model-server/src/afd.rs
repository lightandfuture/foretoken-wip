// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Private worker observations for managed single-rank AFD members.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Json, Router,
    http::StatusCode,
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader, Lines};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;

use crate::api::RuntimeHealth;
use crate::launch::{AfdPlan, ExecutionRole};

/// Owns the socket that receives lifecycle events from this member's only GPU worker.
pub struct Observer {
    path: PathBuf,
    listener: UnixListener,
    input: Mutex<Option<Lines<BufReader<UnixStream>>>>,
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
enum Event {
    ConnectorReady { error: Option<String> },
    LoopReady { error: Option<String> },
    Fatal { error: String },
}

impl Observer {
    /// Binds a unique local endpoint before spawn; Drop removes the owned socket path.
    pub fn bind() -> io::Result<Self> {
        let path = std::env::temp_dir().join(format!("afd-{}.sock", uuid::Uuid::new_v4().simple()));
        Ok(Self {
            listener: UnixListener::bind(&path)?,
            path,
            input: Mutex::new(None),
        })
    }

    /// Passes the worker observation address through the managed child's environment.
    pub fn configure(&self, command: &mut std::process::Command) {
        command.env("FORETOKEN_AFD_STATUS_SOCKET", &self.path);
        if let Ok(plugins) = std::env::var("VLLM_PLUGINS") {
            command.env("VLLM_PLUGINS", format!("{plugins},afd,foretoken_afd"));
        }
    }

    /// Waits for actual connector completion and, for FFN, entry into its receive loop.
    pub async fn ready(&self, ffn: bool) -> io::Result<()> {
        let mut connector = false;
        let mut loop_ready = !ffn;
        while !connector || !loop_ready {
            match self.next().await? {
                Event::ConnectorReady { error: None } => connector = true,
                Event::LoopReady { error: None } => loop_ready = true,
                Event::Fatal { error } => return Err(io::Error::other(error)),
                _ => return Err(io::Error::other("invalid AFD worker readiness event")),
            }
        }
        Ok(())
    }

    /// Watches the same worker connection for fatal errors or loss after initialization.
    pub async fn failure(&self) -> io::Error {
        loop {
            match self.next().await {
                Ok(Event::Fatal { error }) => return io::Error::other(error),
                Err(error) => return error,
                Ok(_) => {}
            }
        }
    }

    /// Reads from the retained worker stream; cancellation preserves partial frame buffers.
    async fn next(&self) -> io::Result<Event> {
        let mut input = self.input.lock().await;
        if input.is_none() {
            *input = Some(BufReader::new(self.listener.accept().await?.0).lines());
        }
        let line = input
            .as_mut()
            .expect("connected observer")
            .next_line()
            .await?
            .ok_or_else(|| io::Error::other("AFD worker observation channel closed"))?;
        serde_json::from_str(&line).map_err(io::Error::other)
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Exposes member observations and the Attention-only serving gate to the Group controller.
/// FFN retains read-only status; neither role gains a generation route here.
pub fn status_router(plan: AfdPlan, health: Arc<RuntimeHealth>) -> Router {
    let mut router = Router::new();
    if plan.role == ExecutionRole::Attention {
        let identity = plan.clone();
        let admission = health.clone();
        router = router.route(
            "/internal/afd/serving",
            post(move |Json(request): Json<ServingRequest>| {
                let plan = identity.clone();
                let health = admission.clone();
                async move {
                    if request.group_uid != plan.group_uid
                        || request.member_index != plan.member_index
                    {
                        return Err(StatusCode::CONFLICT);
                    }
                    health.set_afd_serving(request.enabled);
                    Ok(Json(json!({
                        "groupUid": plan.group_uid,
                        "role": plan.role,
                        "memberIndex": plan.member_index,
                        "memberCount": plan.member_count,
                        "ready": health.ready(),
                        "accepting": health.accepting(),
                    })))
                }
            }),
        );
    }
    router.route(
        "/internal/afd/status",
        get(move || {
            let plan = plan.clone();
            let health = health.clone();
            async move {
                Json(json!({
                    "groupUid": plan.group_uid,
                    "role": plan.role,
                    "memberIndex": plan.member_index,
                    "memberCount": plan.member_count,
                    "ready": health.ready(),
                }))
            }
        }),
    )
}

// Only the existing Group owner controls this gate after observing the complete AF pair.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ServingRequest {
    group_uid: String,
    member_index: usize,
    enabled: bool,
}
