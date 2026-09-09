# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

"""Bind trace schedule events to native or external request payloads."""

from __future__ import annotations

from dataclasses import replace

from benchmarks.config import BenchConfig
from benchmarks.workload.hf_dataset import same_dataset_source
from benchmarks.workload.loader import load_indexed_requests, load_requests
from benchmarks.workload.random_dataset import (
    generate_synthetic_prefix_reuse_requests,
    generate_random_requests,
)
from benchmarks.workload.trace_loader import TraceLoader, TraceRequest


def resolve_payload_source(config: BenchConfig) -> str:
    dataset = config.dataset
    if dataset.dataset == ["random"]:
        return "random"
    return "dataset"


def build_trace_requests(
    config: BenchConfig,
    events: list[TraceRequest],
) -> tuple[str, list[TraceRequest]]:
    """Pair selected trace events with the configured payload source."""
    payload_source = resolve_payload_source(config)
    if payload_source == "random":
        input_lengths = [event.input_length for event in events]
        has_input_lengths = [length is not None for length in input_lengths]
        if any(has_input_lengths) and not all(has_input_lengths):
            raise ValueError(
                "Random trace payload requires input_length on every "
                "selected trace event"
            )
        if config.dataset.trace_synthetic_prefix_reuse:
            if not all(has_input_lengths):
                raise ValueError(
                    "--trace-synthetic-prefix-reuse requires input_length "
                    "on every selected trace event"
                )
            payloads = generate_synthetic_prefix_reuse_requests(
                config,
                input_lengths=[int(length) for length in input_lengths],
                hash_id_lists=[event.hash_ids for event in events],
            )
        else:
            payloads = generate_random_requests(
                config,
                number=len(events),
                input_lengths=(
                    [int(length) for length in input_lengths]
                    if all(has_input_lengths)
                    else None
                ),
            )
    elif same_dataset_source(
        config.dataset.dataset[0], config.dataset.trace_path
    ):
        has_native_payloads = all(
            event.messages is not None or event.prompt is not None
            for event in events
        )
        if has_native_payloads:
            for index, event in enumerate(events):
                events[index] = replace(event, payload_source=payload_source)
            return payload_source, events
        payloads = load_indexed_requests(
            config.dataset.dataset[0],
            [event.source_index for event in events],
        )
    else:
        payloads = load_requests(config, number=len(events))

    if len(payloads) != len(events):
        raise ValueError(
            f"Loaded {len(payloads)} payloads for {len(events)} trace events"
        )

    for index, (event, payload) in enumerate(zip(events, payloads)):
        events[index] = replace(
            event,
            messages=payload.get("messages"),
            prompt=payload.get("prompt"),
            tools=payload.get("tools"),
            payload_source=payload_source,
        )
    return payload_source, events


def load_trace_workload(
    config: BenchConfig,
) -> tuple[float, str, str, list[TraceRequest]]:
    """Load a trace window and bind its events to final request payloads."""
    dataset = config.dataset
    loader = TraceLoader(dataset.trace_path)
    trace_window_start_s, events = loader.load_window(
        start=dataset.trace_start,
        duration=dataset.trace_duration,
    )
    payload_source, requests = build_trace_requests(config, events)
    if loader.trace_format is None:
        raise RuntimeError("Trace format was not detected")
    return trace_window_start_s, loader.trace_format, payload_source, requests
