"""Shared argument parsing for the worker scripts."""
from __future__ import annotations

import argparse


def db_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="warehouse")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="warehouse")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--app", required=True, help="application_name to report")
    return parser


def conninfo(args: argparse.Namespace) -> dict:
    return {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password,
        "application_name": args.app,
        "connect_timeout": 5,
    }
