"""Client for the SimpleTac MessagePack websocket inference protocol."""

from __future__ import annotations

import logging
import time
from typing import Any

import websockets.sync.client

from openpi_client import msgpack_numpy


class WebsocketClientPolicy:
    """Send observations to a remote policy and receive action chunks."""

    def __init__(self, host: str = "0.0.0.0", port: int | None = None) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        logging.info("Waiting for SimpleTac server at %s", self._uri)
        while True:
            try:
                connection = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                )
                metadata = msgpack_numpy.unpackb(connection.recv())
                return connection, metadata
            except (ConnectionRefusedError, OSError):
                logging.info("SimpleTac server is not ready; retrying")
                time.sleep(5)

    def get_server_metadata(self) -> dict[str, Any]:
        """Return metadata sent during the websocket handshake."""
        return self._server_metadata

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Run one synchronous policy request."""
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"SimpleTac inference failed:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        """The server is stateless, so reset is intentionally a no-op."""

    def close(self) -> None:
        """Close the websocket connection."""
        self._ws.close()
