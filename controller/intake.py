"""Block 1 - Intake: turn an IncidentReport into IncidentFeatures.

Three things happen here, all standard log/alert feature engineering:

1. Select evidence: keep only error-ish log lines, mask volatile tokens (timestamps, IPs,
   numbers, hex ids) and de-duplicate on the masked form, so 30 copies of the same failure
   become one line.
2. Extract signals from the *raw* evidence (masking would hide "exit 137" or "Errno 28").
   A signal is an observation ("lock_timeout", "disk_full", ...), never an owner. Mapping
   signals to owning domains is the classifier's job (Block 3), learned from data.
3. Build a compact, masked text for embedding models and LLM prompts.
"""
from __future__ import annotations

import re

from .contracts import IncidentFeatures, IncidentReport

MAX_LINES_PER_SERVICE = 6
MAX_ERROR_LINES = 15
MAX_TEXT_CHARS = 2000

# -- masking -------------------------------------------------------------------------------------

_MASKS = (
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b[0-9a-f]{8,}\b"), "<HEX>"),
    (re.compile(r"\b\d+(?:\.\d+)?"), "<N>"),
)


def mask(text: str) -> str:
    for pattern, token in _MASKS:
        text = pattern.sub(token, text)
    return re.sub(r"\s+", " ", text).strip()


# -- evidence selection ------------------------------------------------------------------------------

_ERRORISH = re.compile(
    r"\b(ERROR|CRITICAL|FATAL|PANIC|WARNING|error|failed|failure|fail|killed|timeout|timed out|refused)\b",
    re.IGNORECASE,
)
_LEADING_STAMP = re.compile(r"^<TS>\s*")


def select_error_lines(logs: dict[str, str]) -> list[tuple[str, str]]:
    """Return (masked "[service] line", raw line) pairs, de-duplicated on the masked form."""
    seen: set[str] = set()
    selected: list[tuple[str, str]] = []
    for service in sorted(logs):
        kept = 0
        for raw in logs[service].splitlines():
            if not _ERRORISH.search(raw):
                continue
            line = _LEADING_STAMP.sub("", mask(raw))
            key = f"[{service}] {line}"
            if line and key not in seen:
                seen.add(key)
                selected.append((key, raw.strip()))
                kept += 1
                if kept >= MAX_LINES_PER_SERVICE:
                    break
    return selected[:MAX_ERROR_LINES]


# -- signals -----------------------------------------------------------------------------------------------
# Observation vocabulary. Patterns match what the platform's services, Postgres, httpx and dbt
# actually print. One line can show several signals.

SIGNAL_PATTERNS: dict[str, re.Pattern] = {name: re.compile(p, re.IGNORECASE) for name, p in {
    "network_timeout": r"\b(?:Read|Connect|Write|Pool)Timeout\b|\btimed out\b",
    "lock_timeout": r"lock timeout|LockNotAvailable|could not obtain lock",
    "statement_timeout": r"statement timeout|QueryCanceled",
    "connection_reset": r"connection reset|reset by peer|server closed the connection unexpectedly",
    "connect_failed": r"connection refused|\bConnectError\b|could not connect|connection to server .* failed",
    "name_resolution": r"name or service not known|name does not resolve|temporary failure in name resolution"
                       r"|could not translate host name|getaddrinfo failed",
    "too_many_clients": r"too many clients|remaining connection slots are reserved",
    "db_saturation": r"warehouse connections at",
    "auth_failed": r"password authentication failed",
    "schema_mismatch": r"SchemaMismatch|missing required field",
    "data_test_failed": r"\bnot_null_|\bunique_|accepted_values|configured to fail|Failure in test",
    "oom_killed": r"exit 137|\bSIGKILL\b|out of memory|OOMKilled",
    "disk_full": r"No space left on device|Errno 28|could not extend file",
    "config_invalid": r"invalid configuration|refusing to start",
    "service_down": r"\bAPI down\b|crash-?loop",
    "http_5xx": r"\bHTTP 5\d\d\b|returned 5\d\d|requests failed",
    "high_latency": r"p95 latency",
    "stale_data": r"behind the source|has no data",
    "login_failures": r"failed (?:dashboard )?logins?",
    "unexpected_db_client": r"unexpected application",
}.items()}


def extract_signals(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        for name, pattern in SIGNAL_PATTERNS.items():
            if pattern.search(line):
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


# -- components ------------------------------------------------------------------------------------------
# Static topology: which part of the platform each alert watches (alert metadata, not ownership).

ALERT_COMPONENTS = {
    "OrdersMartStale": "marts",
    "IngestJobFailing": "ingest",
    "TransformJobFailing": "transform",
    "DashboardDown": "metrics-api",
    "DashboardSlow": "metrics-api",
    "LoginFailureSpike": "metrics-api",
    "WarehouseConnectionsSaturated": "warehouse",
    "UnexpectedWarehouseClient": "warehouse",
    "MonitoringGap": "monitoring",
}


def affected_components(report: IncidentReport, error_lines: list[str]) -> list[str]:
    components = {ALERT_COMPONENTS.get(a.name, "unknown") for a in report.alerts}
    components |= {line[1:line.index("]")] for line in error_lines}
    return sorted(components)


# -- text ----------------------------------------------------------------------------------------------------

def build_text(report: IncidentReport, error_lines: list[str]) -> str:
    parts = [f"title: {report.title}", "alerts:"]
    parts += [f"- {a.name}: {mask(a.detail)}" for a in report.alerts]
    if error_lines:
        parts.append("evidence:")
        parts += [f"- {line}" for line in error_lines]
    return "\n".join(parts)[:MAX_TEXT_CHARS]


def extract(report: IncidentReport) -> IncidentFeatures:
    selected = select_error_lines(report.logs)
    error_lines = [masked for masked, _ in selected]
    job_errors: dict[str, str] = {}  # masked -> raw, de-duplicated
    for run in report.recent_job_runs:
        if run.error:
            job_errors.setdefault(f"[{run.job} job] {mask(run.error)}", run.error)
    raw_evidence = [a.detail for a in report.alerts] + [raw for _, raw in selected] + list(job_errors.values())
    return IncidentFeatures(
        incident_id=report.incident_id,
        alerts=[a.name for a in report.alerts],
        signals=extract_signals(raw_evidence),
        components=affected_components(report, error_lines),
        error_lines=error_lines,
        text=build_text(report, error_lines),
    )
