# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project


"""Immutable run context for one benchmark point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from benchmarks.config import BenchConfig


@dataclass(frozen=True)
class RunSpec:
    """Resolved point: config plus orchestration context from a parent runner."""

    config: BenchConfig
    label: str = ""
    output_dir: Optional[str] = None
    wandb_group: Optional[str] = None
