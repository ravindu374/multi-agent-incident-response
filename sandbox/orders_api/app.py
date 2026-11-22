"""Simulated source system: the food-delivery orders API.

Orders are a pure function of their id (seeded RNG), and order N "exists" once
N / ORDERS_PER_SEC seconds have passed since the stack's epoch. The service is
stateless apart from the epoch file, so restarts never lose or reorder data.

/admin/* is the lab's fault switchboard. Agents never get access to it.
"""
from __future__ import annotations

import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel

RATE = float(os.getenv("ORDERS_PER_SEC", "5"))
EPOCH_FILE = Path(os.getenv("EPOCH_FILE", "/data/epoch"))
STATUSES = ("placed", "delivered", "delivered", "delivered", "cancelled", "refunded")


def _load_epoch() -> float:
    try:
        return float(EPOCH_FILE.read_text())
    except (FileNotFoundError, ValueError):
        now = time.time()
        EPOCH_FILE.parent.mkdir(parents=True, exist_ok=True)
        EPOCH_FILE.write_text(str(now))
        return now


EPOCH = _load_epoch()
app = FastAPI(title="orders-api")

_lock = threading.Lock()
_faults: dict[str, int | bool] = {"schema_drift": False, "bad_rows_batches": 0}


def max_available() -> int:
    return int((time.time() - EPOCH) * RATE)


def make_order(order_id: int) -> dict:
    rng = random.Random(order_id)
    created = datetime.fromtimestamp(EPOCH + order_id / RATE, tz=timezone.utc)
    return {
        "order_id": order_id,
        "created_at": created.isoformat(),
        "customer_id": rng.randint(1, 5000),
        "store_id": rng.randint(1, 20),
        "amount": round(rng.lognormvariate(3.0, 0.5), 2),
        "status": rng.choice(STATUSES),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/orders")
def orders(after_id: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000)) -> dict:
    top = min(after_id + limit, max_available())
    rows = [make_order(i) for i in range(after_id + 1, top + 1)]

    with _lock:
        drift = bool(_faults["schema_drift"])
        corrupt = bool(rows) and int(_faults["bad_rows_batches"]) > 0
        if corrupt:
            _faults["bad_rows_batches"] = int(_faults["bad_rows_batches"]) - 1

    if corrupt:
        for row in rows[:3]:
            row["amount"] = None
        if after_id > 1:  # the source "re-sends" orders it already delivered
            rows.append(make_order(after_id))
            rows.append(make_order(after_id - 1))
    if drift:
        for row in rows:
            row["total_amount"] = row.pop("amount")

    return {"orders": rows, "max_available": max_available()}


class FaultFlags(BaseModel):
    schema_drift: bool | None = None
    bad_rows_batches: int | None = None


@app.get("/admin/faults")
def get_faults() -> dict:
    with _lock:
        return dict(_faults)


@app.post("/admin/faults")
def set_faults(flags: FaultFlags) -> dict:
    with _lock:
        _faults.update(flags.model_dump(exclude_none=True))
        return dict(_faults)


@app.delete("/admin/faults")
def clear_faults() -> dict:
    with _lock:
        _faults.update({"schema_drift": False, "bad_rows_batches": 0})
        return dict(_faults)
