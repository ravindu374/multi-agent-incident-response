"""Reporting worker that opens connections and never gives them back (D4)."""
from __future__ import annotations

import time

import psycopg

from common import conninfo, db_parser

parser = db_parser(__doc__)
parser.add_argument("--max", type=int, default=60)
args = parser.parse_args()

held = []
for _ in range(args.max):
    try:
        held.append(psycopg.connect(**conninfo(args)))
    except psycopg.OperationalError as exc:
        print(f"connect failed after {len(held)} connections: {str(exc).strip().splitlines()[-1]}", flush=True)
        break
print(f"report worker holding {len(held)} connections", flush=True)
while True:
    time.sleep(3600)
