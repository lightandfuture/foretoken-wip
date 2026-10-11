# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

"""Observe the pinned AFD worker lifecycle without replacing engine execution."""

import functools
import json
import os
import socket
import threading
from datetime import timedelta

_connection = None
_lock = threading.Lock()
_registered = False


def _publish(event: str, error: str | None = None) -> None:
    """Send worker lifecycle facts to the owning model-server over its private socket."""
    global _connection
    with _lock:
        if _connection is None:
            _connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _connection.connect(os.environ["FORETOKEN_AFD_STATUS_SOCKET"])
        _connection.sendall((json.dumps({"event": event, "error": error}) + "\n").encode())


def register() -> None:
    """Install observation hooks in vLLM plugin-loading processes only for managed AFD."""
    global _registered
    if _registered or "FORETOKEN_AFD_STATUS_SOCKET" not in os.environ:
        return
    _registered = True
    from afd_plugin.connectors.gpu.p2p import (
        P2pNcclAFDConnector,
        P2pNcclAFDControlPlane,
    )
    from afd_plugin.v1.worker.ffn_worker import AFDFFNWorker

    # These methods belong to plugin b7561d7. Observe successful collectives and
    # entry into the actual receive loop; Thread.start() is not loop readiness.
    original_init = P2pNcclAFDConnector.init_afd_connector

    @functools.wraps(original_init)
    def initialize(self):
        try:
            original_init(self)
            if self.p2p_pg is None:
                raise RuntimeError("managed 1A1F requires both ranks in the P2P metadata group")
            if self.afd_config.role == "attention":
                # FFN's native handshake precedes its busy loop. It can complete
                # independently while Attention waits in connector initialization.
                # Use the existing group store; no second rendezvous owner is needed.
                key = f"foretoken/{os.environ['FORETOKEN_AFD_GROUP_UID']}/ffn-loop-ready"
                self.p2p_pg.get_group_store().wait(
                    [key], timedelta(seconds=self.afd_config.afd_process_group_timeout_s)
                )
        except Exception as error:
            _publish("fatal", str(error))
            raise
        _publish("connector_ready")

    P2pNcclAFDConnector.init_afd_connector = initialize
    original_receive = P2pNcclAFDControlPlane.recv_dp_metadata_list

    @functools.wraps(original_receive)
    def receive(self, *args, **kwargs):
        if not getattr(self, "_foretoken_loop_reported", False):
            # The FFN worker has set its CUDA device and is entering its first
            # blocking receive. Publish only here, never after Thread.start().
            key = f"foretoken/{os.environ['FORETOKEN_AFD_GROUP_UID']}/ffn-loop-ready"
            self.connector.p2p_pg.get_group_store().set(key, "ready")
            _publish("loop_ready")
            self._foretoken_loop_reported = True
        return original_receive(self, *args, **kwargs)

    P2pNcclAFDControlPlane.recv_dp_metadata_list = receive
    original_check = AFDFFNWorker.raise_ffn_loop_error_if_any

    @functools.wraps(original_check)
    def check_error(self):
        try:
            original_check(self)
        except RuntimeError as error:
            _publish("fatal", str(error.__cause__ or error))
            raise

    AFDFFNWorker.raise_ffn_loop_error_if_any = check_error
