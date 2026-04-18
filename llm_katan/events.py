"""
Event broadcasting for the live dashboard.

Providers call `broadcast()` to send request/response events to all
connected WebSocket clients.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Manages WebSocket connections and broadcasts events."""

    def __init__(self):
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)
        logger.info("Dashboard client connected (%d total)", len(self._clients))

    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)
        logger.info("Dashboard client disconnected (%d total)", len(self._clients))

    async def broadcast(self, event: dict):
        """Send an event to all connected dashboard clients."""
        if not self._clients:
            return
        data = json.dumps(event)
        disconnected = []
        for ws in self._clients:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._clients.discard(ws)


# Global broadcaster instance
broadcaster = EventBroadcaster()


def make_event(
    provider: str,
    method: str,
    path: str,
    status_code: int,
    client_ip: str,
    latency_ms: int | None = None,
    request_headers: dict | None = None,
    request_body: dict | None = None,
    response_body: dict | None = None,
) -> dict:
    """Build a dashboard event dict."""
    return {
        "timestamp": time.time() * 1000,  # JS milliseconds
        "provider": provider,
        "method": method,
        "path": path,
        "status_code": status_code,
        "client_ip": client_ip,
        "latency_ms": latency_ms,
        "request_headers": request_headers,
        "request_body": request_body,
        "response_body": response_body,
    }
