// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Private worker observations for every local rank of an AFD member.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use axum::{
    Json, Router,
    http::StatusCode,
    routing::{get, post},
};
use futures::{Stream, StreamExt, stream::SelectAll};
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::UnixListener;
use tokio::sync::watch;

use crate::api::RuntimeHealth;
use crate::launch::{AfdPlan, ExecutionRole};

/// Owns the local observation socket and aggregation task, never the worker processes.
pub struct Observer {
    path: PathBuf,
    state: watch::Receiver<WorkerState>,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone, Default)]
struct WorkerState {
    ready: bool,
    failure: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    rank: usize,
    event: EventKind,
    error: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum EventKind {
    ConnectorReady,
    WorkerReady,
    LoopReady,
    Fatal,
}

type WorkerStream = Pin<Box<dyn Stream<Item = io::Result<(usize, Event)>> + Send>>;

impl Observer {
    /// Starts exact-set observation before spawn; Drop aborts it and removes the socket.
    pub fn bind(plan: &AfdPlan) -> io::Result<Self> {
        let path = std::env::temp_dir().join(format!("afd-{}.sock", uuid::Uuid::new_v4().simple()));
        let listener = UnixListener::bind(&path)?;
        let (sender, state) = watch::channel(WorkerState::default());
        let plan = plan.clone();
        let task = tokio::spawn(async move {
            let error = observe_workers(listener, plan, &sender).await;
            sender.send_replace(WorkerState {
                ready: false,
                failure: Some(error.to_string()),
            });
        });
        Ok(Self { path, state, task })
    }

    /// Passes the existing private observation channel through the managed child's environment.
    pub fn configure(&self, command: &mut std::process::Command) {
        command.env("FORETOKEN_AFD_STATUS_SOCKET", &self.path);
        if let Ok(plugins) = std::env::var("VLLM_PLUGINS") {
            command.env("VLLM_PLUGINS", format!("{plugins},afd,foretoken_afd"));
        }
    }

    /// Requires connector and execution readiness from every expected local role rank.
    pub async fn ready(&self) -> io::Result<()> {
        let mut state = self.state.clone();
        let value = state
            .wait_for(|state| state.ready || state.failure.is_some())
            .await
            .map_err(|_| io::Error::other("AFD observation task ended"))?;
        match &value.failure {
            Some(error) => Err(io::Error::other(error.clone())),
            None => Ok(()),
        }
    }

    /// Reports any worker fatal or channel loss, including failures observed during startup.
    pub async fn failure(&self) -> io::Error {
        let mut state = self.state.clone();
        match state.wait_for(|state| state.failure.is_some()).await {
            Ok(value) => io::Error::other(value.failure.clone().expect("observed failure")),
            Err(_) => io::Error::other("AFD observation task ended"),
        }
    }
}

/// Aggregates the existing worker protocol and keeps every channel live after readiness.
async fn observe_workers(
    listener: UnixListener,
    plan: AfdPlan,
    state: &watch::Sender<WorkerState>,
) -> io::Error {
    let expected = plan.local_ranks();
    let mut streams = SelectAll::<WorkerStream>::new();
    let mut identities = BTreeMap::new();
    let mut connectors = BTreeSet::new();
    let mut execution = BTreeSet::new();
    let mut connection_id = 0;
    loop {
        let (connection, event) = tokio::select! {
            result = listener.accept() => {
                let (stream, _) = match result {
                    Ok(connection) => connection,
                    Err(error) => return error,
                };
                let id = connection_id;
                connection_id += 1;
                streams.push(Box::pin(async_stream::try_stream! {
                    let mut lines = BufReader::new(stream).lines();
                    while let Some(line) = lines.next_line().await? {
                        let event = serde_json::from_str::<Event>(&line).map_err(io::Error::other)?;
                        yield (id, event);
                    }
                    Err(io::Error::other(format!("AFD worker channel {id} closed")))?;
                }));
                continue;
            },
            event = streams.next(), if !streams.is_empty() => match event {
                Some(Ok(event)) => event,
                Some(Err(error)) => return error,
                None => return io::Error::other("AFD worker channels ended"),
            },
        };
        // One connection owns exactly one expected worker; neither duplicates nor
        // identities changing midstream may stand in for a missing local rank.
        if !expected.contains(&event.rank)
            || identities
                .iter()
                .any(|(rank, id)| (*rank == event.rank) != (*id == connection))
        {
            return io::Error::other(format!(
                "invalid AFD worker identity {} on channel {connection}",
                event.rank
            ));
        }
        identities.insert(event.rank, connection);
        if let Some(error) = event.error {
            return io::Error::other(format!("AFD worker {}: {error}", event.rank));
        }
        let inserted = match event.event {
            EventKind::ConnectorReady => connectors.insert(event.rank),
            EventKind::LoopReady
                if plan.role == ExecutionRole::Ffn && connectors.contains(&event.rank) =>
            {
                execution.insert(event.rank)
            }
            EventKind::WorkerReady
                if plan.role == ExecutionRole::Attention && connectors.contains(&event.rank) =>
            {
                execution.insert(event.rank)
            }
            _ => false,
        };
        if !inserted {
            return io::Error::other(format!(
                "invalid AFD readiness event for worker {}",
                event.rank
            ));
        }
        if connectors.len() == expected.len() && execution.len() == expected.len() {
            state.send_modify(|state| state.ready = true);
        }
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        self.task.abort();
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Exposes member observations and the Attention-only serving gate to the Group controller.
/// FFN retains read-only status; neither role gains a generation route here.
pub fn status_router(plan: AfdPlan, health: Arc<RuntimeHealth>) -> Router {
    let mut router = Router::new();
    if plan.role == ExecutionRole::Attention && plan.member_index == 0 {
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
