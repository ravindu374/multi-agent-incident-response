"""Minimal Toxiproxy admin client. Lab-only: agents measure the network, they never read this API."""
from __future__ import annotations

import httpx


class Toxiproxy:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def add_toxic(self, proxy: str, name: str, toxic_type: str, attributes: dict,
                  stream: str = "downstream", toxicity: float = 1.0) -> None:
        self.remove_toxic(proxy, name)
        resp = httpx.post(
            f"{self.base_url}/proxies/{proxy}/toxics",
            json={"name": name, "type": toxic_type, "stream": stream, "toxicity": toxicity,
                  "attributes": attributes},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def remove_toxic(self, proxy: str, name: str) -> None:
        resp = httpx.delete(f"{self.base_url}/proxies/{proxy}/toxics/{name}", timeout=self.timeout)
        if resp.status_code not in (200, 204, 404):
            resp.raise_for_status()

    def reset(self) -> None:
        """Remove every toxic and re-enable every proxy."""
        httpx.post(f"{self.base_url}/reset", timeout=self.timeout).raise_for_status()
