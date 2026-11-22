"""Payload validation for the ingest job (no I/O, so it can be unit-tested on the host)."""
from __future__ import annotations

REQUIRED_FIELDS = ("order_id", "created_at", "customer_id", "store_id", "amount", "status")


class SchemaMismatch(ValueError):
    """The source payload no longer matches the warehouse contract."""


def normalise(rows: list[dict], field_map: dict[str, str]) -> list[dict]:
    """Rename source fields via field_map and check every required field is present."""
    out = []
    for raw in rows:
        row = {field_map.get(key, key): value for key, value in raw.items()}
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise SchemaMismatch(
                f"source payload missing required field(s) {missing}; got {sorted(raw)}"
            )
        out.append(row)
    return out
