"""Client hammering the dashboard login with guessed passwords (S2)."""
from __future__ import annotations

import argparse
import random
import secrets
import threading
import time

import httpx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://metrics-api:8000")
parser.add_argument("--threads", type=int, default=16)
args = parser.parse_args()

USERS = ("admin", "analyst", "root", "ops", "finance")
attempts = 0
lock = threading.Lock()


def worker() -> None:
    global attempts
    with httpx.Client(base_url=args.url, timeout=10) as client:
        while True:
            try:
                client.post("/api/login", json={"username": random.choice(USERS), "password": secrets.token_hex(4)})
            except httpx.HTTPError:
                time.sleep(0.5)
            with lock:
                attempts += 1
                if attempts % 500 == 0:
                    print(f"{attempts} attempts", flush=True)


for _ in range(args.threads):
    threading.Thread(target=worker, daemon=True).start()
while True:
    time.sleep(3600)
