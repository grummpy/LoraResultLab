"""The only module allowed to talk to ComfyUI.

Every outbound call is refused unless the settings toggle is on and the
request carries confirm=True. History reads are GET only. Nothing here runs
on import, and the default host is blank so a fresh install cannot connect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.settings import Settings


class ComfyGuardError(PermissionError):
    """Raised before any socket is opened."""


@dataclass
class RecordingTransport:
    """Test double. Records calls and does not touch the network."""

    calls: list = field(default_factory=list)

    def request(self, method: str, url: str, json: dict | None = None):
        self.calls.append({"method": method, "url": url, "json": json})
        return 200, "{}"


class HttpxTransport:
    """Real HTTP. Used only after the guard checks pass."""

    def request(self, method: str, url: str, json: dict | None = None):
        import httpx

        with httpx.Client(timeout=15.0) as client:
            response = client.request(method, url, json=json)
        return response.status_code, response.text


def normalize_host(host: str) -> str:
    text = (host or "").strip()
    if not text:
        raise ComfyGuardError("ComfyUI host is blank. No request was sent.")
    if "://" not in text:
        text = "http://" + text
    return text.rstrip("/")


def assert_comfy_allowed(settings: Settings, confirm: object) -> str:
    """Shared gate for every ComfyUI HTTP call. Returns the normalized host."""
    if confirm is not True:
        raise ComfyGuardError(
            "Confirmation is required before any ComfyUI request. "
            "No request was sent."
        )
    if not settings.comfyui_enabled:
        raise ComfyGuardError("ComfyUI is off. No request was sent.")
    return normalize_host(settings.comfyui_host)


def submit_workflows(
    settings: Settings,
    workflows: list[dict],
    *,
    confirm: object,
    transport,
) -> dict:
    """POST /prompt for each workflow. Guard runs before the transport is used."""
    host = assert_comfy_allowed(settings, confirm)
    sent = []
    for workflow in workflows:
        url = host + "/prompt"
        status, body = transport.request("POST", url, json={"prompt": workflow})
        sent.append({"url": url, "status": status, "body": body[:500]})
    return {
        "submitted": True,
        "comfyui_called": True,
        "method": "POST",
        "count": len(sent),
        "results": sent,
        "warning": "Jobs were sent to ComfyUI and may use Zach's PC GPU.",
    }


def fetch_history(settings: Settings, *, confirm: object, user_initiated: object, transport) -> dict:
    """GET /history only. POST is never used, and the guard still applies."""
    if user_initiated is not True:
        raise ComfyGuardError(
            "History is fetched only when you click Fetch. No request was sent."
        )
    host = assert_comfy_allowed(settings, confirm)
    url = host + "/history"
    status, body = transport.request("GET", url, json=None)
    return {
        "comfyui_called": True,
        "method": "GET",
        "url": url,
        "status": status,
        "body": body,
        "submitted": False,
    }
