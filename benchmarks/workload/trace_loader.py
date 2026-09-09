# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

"""Load supported conversation traces as timestamped replay requests."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from benchmarks.workload.hf_dataset import (
    is_hf_dataset_spec,
    iter_hf_rows,
    resolve_hf_file_uri,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceRequest:
    """One normalized trace event with an optional bound payload."""

    timestamp_s: float
    source_index: int
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    input_length: int | None = None
    hash_ids: list[int] | None = None
    conversation_id: str | None = None
    payload_source: str = ""


class StudyChatAdapter:
    """Convert StudyChat JSONL rows into ``TraceRequest`` objects."""

    @staticmethod
    def parse(
        payload: object,
        *,
        path: Path,
        line_no: int,
        source_index: int,
    ) -> TraceRequest:
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {path}:{line_no}")

        required = ("timestamp", "chatId", "messages")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                f"Missing {', '.join(missing)} at {path}:{line_no}"
            )

        try:
            timestamp_ms = float(payload["timestamp"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid timestamp at {path}:{line_no}") from error
        if not math.isfinite(timestamp_ms):
            raise ValueError(f"Timestamp must be finite at {path}:{line_no}")

        chat_id = payload["chatId"]
        if chat_id is None or not str(chat_id):
            raise ValueError(f"Empty chatId at {path}:{line_no}")

        messages = payload["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Invalid messages at {path}:{line_no}")

        input_length = payload.get("input_length")
        if input_length is not None:
            try:
                input_length = int(input_length)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid input_length at {path}:{line_no}"
                ) from error
            if input_length <= 0:
                raise ValueError(f"Invalid input_length at {path}:{line_no}")

        return TraceRequest(
            timestamp_s=timestamp_ms / 1000.0,
            source_index=source_index,
            messages=messages,
            input_length=input_length,
            conversation_id=str(chat_id),
        )


class MooncakeAdapter:
    """Convert Mooncake JSONL rows into timestamped trace events."""

    @staticmethod
    def parse(
        payload: object,
        *,
        path: Path,
        line_no: int,
        source_index: int,
    ) -> TraceRequest:
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {path}:{line_no}")
        if "timestamp" not in payload or "input_length" not in payload:
            raise ValueError(
                f"Mooncake trace needs timestamp and input_length at "
                f"{path}:{line_no}"
            )
        try:
            timestamp_ms = float(payload["timestamp"])
            input_length = int(payload["input_length"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid timestamp or input_length at {path}:{line_no}"
            ) from error
        if not math.isfinite(timestamp_ms) or input_length <= 0:
            raise ValueError(
                f"Invalid timestamp or input_length at {path}:{line_no}"
            )

        hash_ids = payload.get("hash_ids")
        if hash_ids is not None:
            if not isinstance(hash_ids, list) or any(
                isinstance(hash_id, bool)
                or not isinstance(hash_id, int)
                or hash_id < 0
                for hash_id in hash_ids
            ):
                raise ValueError(f"Invalid hash_ids at {path}:{line_no}")

        conversation_id = payload.get("chatId")
        return TraceRequest(
            timestamp_s=timestamp_ms / 1000.0,
            source_index=source_index,
            input_length=input_length,
            hash_ids=hash_ids,
            conversation_id=(
                str(conversation_id) if conversation_id is not None else None
            ),
        )


class TraceLoader:
    """Load and stably order requests from a supported trace format."""

    _ADAPTERS = {
        "studychat": StudyChatAdapter,
        "mooncake": MooncakeAdapter,
    }

    def __init__(self, path: str | Path):
        self.trace_path = str(path)
        self.trace_format: str | None = None

    def _resolve_path(self) -> Path:
        """Resolve a local path, HF dataset file, or known trace source."""
        local = Path(self.trace_path).expanduser()
        if local.exists():
            return local

        if not self.trace_path.startswith("hf://"):
            return local

        logger.info("Resolving Hugging Face trace %s", self.trace_path)
        return Path(resolve_hf_file_uri(self.trace_path))

    def _iter_rows(self) -> Iterator[tuple[Path, int, int, Any]]:
        """Yield ``(label, line_no, source_index, payload)`` rows."""
        local = Path(self.trace_path).expanduser()
        is_hf_dataset = (
            not local.exists()
            and not self.trace_path.startswith("hf://")
            and is_hf_dataset_spec(self.trace_path)
        )
        if is_hf_dataset:
            label = Path(f"hf://{self.trace_path}")
            logger.info("Resolving Hugging Face trace %s", self.trace_path)
            for row_index, payload in iter_hf_rows(self.trace_path):
                yield label, row_index + 1, row_index, payload
            return

        path = self._resolve_path()
        if not path.is_file():
            raise FileNotFoundError(f"Trace JSONL not found: {path}")
        source_index = 0
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    payload: Any = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_no}: {error}"
                    ) from error
                yield path, line_no, source_index, payload
                source_index += 1

    def _iter_requests(self) -> Iterator[TraceRequest]:
        """Parse trace rows without retaining the complete source in memory."""
        adapter = None
        for path, line_no, source_index, payload in self._iter_rows():
            if adapter is None:
                trace_format = self._detect_format(
                    payload,
                    path=path,
                    line_no=line_no,
                )
                if self.trace_format not in (None, trace_format):
                    raise ValueError("Trace format changed between reads")
                self.trace_format = trace_format
                adapter = self._ADAPTERS[trace_format]
            yield adapter.parse(
                payload,
                path=path,
                line_no=line_no,
                source_index=source_index,
            )

    @staticmethod
    def _detect_format(
        payload: object,
        *,
        path: Path,
        line_no: int,
    ) -> str:
        if isinstance(payload, dict):
            if all(
                key in payload
                for key in ("timestamp", "chatId", "messages")
            ):
                return "studychat"
            if all(key in payload for key in ("timestamp", "input_length")):
                return "mooncake"
        raise ValueError(
            f"Cannot detect trace format from {path}:{line_no}; expected "
            "StudyChat timestamp/chatId/messages or Mooncake "
            "timestamp/input_length"
        )

    def load_window(
        self,
        *,
        start: float = 0.0,
        duration: float | None = None,
    ) -> tuple[float, list[TraceRequest]]:
        self.trace_format = None
        first_timestamp = None
        for request in self._iter_requests():
            if first_timestamp is None:
                first_timestamp = request.timestamp_s
            else:
                first_timestamp = min(first_timestamp, request.timestamp_s)

        if first_timestamp is None:
            raise ValueError("Trace contains no requests")

        window_start = first_timestamp + start
        window_end = (
            None if duration is None else window_start + duration
        )
        selected = [
            request
            for request in self._iter_requests()
            if request.timestamp_s >= window_start
            and (window_end is None or request.timestamp_s < window_end)
        ]
        if not selected:
            raise ValueError("Trace window contains no requests")
        selected.sort(key=lambda request: request.timestamp_s)
        return window_start, selected
