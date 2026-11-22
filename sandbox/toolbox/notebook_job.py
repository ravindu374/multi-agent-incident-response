"""Analyst notebook running an expensive query in a loop (D5). A worker stops when its session is killed."""
from __future__ import annotations

import threading

import psycopg

from common import conninfo, db_parser

QUERY = (
    "SELECT sum(length(md5(o.order_id::text || g::text))) "
    "FROM raw.orders o CROSS JOIN generate_series(1, %s) g"
)

parser = db_parser(__doc__)
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--fanout", type=int, default=300)
args = parser.parse_args()


def worker(index: int) -> None:
    try:
        with psycopg.connect(**conninfo(args), autocommit=True) as conn:
            while True:
                conn.execute(QUERY, (args.fanout,))
    except psycopg.Error as exc:
        print(f"worker {index} stopped: {type(exc).__name__}: {str(exc).strip()[:200]}", flush=True)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.threads)]
for t in threads:
    t.start()
for t in threads:
    t.join()
