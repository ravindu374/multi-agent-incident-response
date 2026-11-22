"""Unapproved client bulk-copying raw orders with borrowed credentials (S1). Retries with backoff."""
from __future__ import annotations

import threading
import time

import psycopg

from common import conninfo, db_parser

parser = db_parser(__doc__)
parser.add_argument("--threads", type=int, default=6)
args = parser.parse_args()


def worker(index: int) -> None:
    backoff = 1.0
    while True:
        try:
            with psycopg.connect(**conninfo(args)) as conn:
                backoff = 1.0
                with conn.cursor() as cur:
                    with cur.copy("COPY (SELECT * FROM raw.orders) TO STDOUT") as copy:
                        for _ in copy:
                            pass
        except psycopg.Error as exc:
            print(f"worker {index}: {type(exc).__name__}: {str(exc).strip()[:160]}", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


for i in range(args.threads):
    threading.Thread(target=worker, args=(i,), daemon=True).start()
while True:
    time.sleep(3600)
