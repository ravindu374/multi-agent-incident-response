"""Is a proposed action something an executor could run, and is it backed by evidence?

Two checks, shared by the expert agents (before a proposal leaves), the agent evaluation, and the
policy agent (Block 5c):

  executable  the target is a platform component, and the action names what it acts on
              (stop_session needs a pid / application / user / client, block_client an ip, ...)
  grounded    every identifier in it (pid, ip, application name, user, role, mount, renamed field)
              appears in the evidence the agent actually saw: its tool results or the incident.
              This catches invented ids. It cannot tell a culprit from a victim (both appear in
              the lock table); that needs ground truth, so the evaluation checks it separately.
  located     an id must be found where the action acts: a process to stop must be in the process
              list OF THE TARGET component (pid 5118 in the warehouse is not a transform process), and a
              field to map must be one the source actually sends (source_sample), which catches a
              mapping written backwards ({"amount": "total_amount"} instead of {"total_amount": "amount"}).
              A process to stop must also not be the component itself: never pid 1 (its main process),
              never one of the database's own "postgres" processes (see OWN_PROCESSES).
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic_core import to_jsonable_python

from .plan import COMPONENTS, INTENTS, PlannedAction

# params that name a concrete thing in the system, so they must come from the evidence
GROUNDED_PARAMS = ("pid", "pids", "application_name", "usename", "client_addr", "ip", "role", "mount", "mapping")
CHECK = "action-check"  # Finding.tool for the notes these checks leave in a proposal

# A component's own server processes, by command prefix: stop_process must never target them. In the
# warehouse every "postgres ..." process is the database itself (pid 1 the server, "postgres: checkpointer"
# its helpers, "postgres: analyst warehouse ... SELECT" a session's backend), and stop_process is a
# `kill -9`: killing any of them makes Postgres drop every session and run crash recovery. A runaway
# query is ended with stop_session (data domain) instead. Both LLMs did this in the first runs: qwen and
# llama3 listed every pid of the warehouse, pid 1 included, and llama3 tried to kill a notebook's queries.
OWN_PROCESSES = {"warehouse": ("postgres",)}


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _changes_nothing(value: Any) -> bool:
    """A mapping whose every field maps onto itself ({"amount": "amount"}) renames nothing."""
    return isinstance(value, dict) and bool(value) and all(str(k) == str(v) for k, v in value.items())


def _scalar(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def identifiers(action: PlannedAction) -> dict[str, list[str]]:
    """The action's identifying values as strings, keyed by param (pid and pids merged into 'pid',
    a mapping contributes its source field names, 'target' is the component)."""
    ids: dict[str, list[str]] = {"target": [action.target]}
    for key, value in action.params.items():
        if _empty(value):
            continue
        name = "pid" if key in ("pid", "pids") else key
        if isinstance(value, dict):  # a mapping: the fields it renames ("a -> a" changes nothing)
            values = [str(k) for k, v in value.items() if str(k) != str(v)]
            if not values:
                continue
        elif isinstance(value, (list, tuple)):
            values = [_scalar(v) for v in value]
        else:
            values = [_scalar(value)]
        ids.setdefault(name, []).extend(values)
    return ids


def same_fix(a: list[PlannedAction], b: list[PlannedAction]) -> bool:
    """Do two fix lists act on the same thing? Same intent, and EVERY kind of identifier both name overlaps
    (pid, ip, renamed field, route AND endpoint, ...), or no identifier at all and the same target
    (restart_component, free_storage). Identifiers of different kinds cannot be compared: "session 556" and
    "the reporting-worker sessions" may be different sessions, so they are not the same fix. (Until Block 6a
    they counted as the same because both act on the warehouse; until the first LLM memory, one overlapping
    kind was enough, so "reroute the warehouse path to secondary" matched "reroute the source path to
    secondary" through "secondary" alone, and the memory credited a wrong fix.)"""
    for x in a:
        for y in b:
            if x.intent != y.intent:
                continue
            ix, iy = identifiers(x), identifiers(y)
            keys = (set(ix) & set(iy)) - {"target"}
            if keys:
                if all(set(ix[k]) & set(iy[k]) for k in keys):
                    return True
            elif len(ix) == len(iy) == 1 and x.target == y.target:  # neither names anything but the target
                return True
    return False


def executable_problems(action: PlannedAction) -> list[str]:
    problems, spec = [], INTENTS[action.intent]
    if action.target not in COMPONENTS:
        problems.append(f"target {action.target!r} is not a platform component")
    elif action.target not in spec.targets:
        problems.append(f"{action.intent} acts on {' or '.join(spec.targets)}, not {action.target}")
    for group in spec.needs:
        values = [action.params.get(k) for k in group]
        if all(_empty(v) for v in values):
            problems.append(f"{action.intent} must say which {' or '.join(group)}")
        elif all(_empty(v) or _changes_nothing(v) for v in values):
            problems.append(f"{action.intent} changes nothing: every field is mapped onto itself")
    return problems


def evidence_text(*parts: Any) -> str:
    """Everything an agent saw, as one searchable text (tool results, incident report, ...)."""
    return "\n".join(json.dumps(to_jsonable_python(p), default=str) for p in parts)


def _mentioned(value: str, text: str) -> bool:
    # whole token only: pid 60 must not match 6079, 172.18.0.5 must not match 172.18.0.55
    return re.search(rf"(?<![\w.\-/]){re.escape(value)}(?![\w.\-])", text) is not None


def grounding_problems(action: PlannedAction, text: str) -> list[str]:
    ids = identifiers(action)
    missing = [f"{key}={value}" for key in dict.fromkeys("pid" if k == "pids" else k for k in GROUNDED_PARAMS)
               for value in ids.get(key, []) if not _mentioned(value, text)]
    return [f"not in the evidence: {', '.join(missing)}"] if missing else []


def _pid_of(process: Any) -> Any:
    return process.get("pid") if isinstance(process, dict) else getattr(process, "pid", None)


def _command_of(process: Any) -> str:
    return str((process.get("command") if isinstance(process, dict) else getattr(process, "command", "")) or "")


def location_problems(action: PlannedAction, results: dict[str, Any]) -> list[str]:
    """Structured check on the agent's tool results: is each id where the action says it is?"""
    if action.intent == "map_schema_fields":
        fields = results.get("source_sample")
        if not fields:
            return ["map_schema_fields needs the fields the source sends (source_sample) as evidence"]
        sent = {str(f) for f in fields}
        stray = [f for f in identifiers(action).get("mapping", []) if f not in sent]
        return [f"field(s) {', '.join(stray)} are not sent by the source (it sends {', '.join(sorted(sent))}); "
                f"a mapping goes from the field the source sends to the field the pipeline expects"] if stray else []
    if action.intent != "stop_process":
        return []
    lists = results.get("top_processes")
    if not lists:
        return ["stop_process needs the target's process list (top_processes) as evidence"]
    listed = {str(_pid_of(p)): _command_of(p) for p in (lists.get(action.target) or [])}
    pids = identifiers(action).get("pid", [])
    problems = []
    stray = [pid for pid in pids if pid not in listed]
    if stray:
        problems.append(f"pid(s) {', '.join(stray)} are not in {action.target}'s process list")
    if "1" in pids:
        problems.append(f"pid 1 is {action.target}'s main process: stopping it stops {action.target} "
                        f"(restart_component restarts it)")
    own = [pid for pid in pids if pid != "1" and pid in listed
           and listed[pid].startswith(OWN_PROCESSES.get(action.target, ()))]
    if own:
        problems.append(f"pid(s) {', '.join(own)} are {action.target}'s own database processes "
                        f"('{listed[own[0]]}'): killing one makes the database drop every session; "
                        f"a query or session is ended with stop_session (data domain)")
    return problems


def action_problems(action: PlannedAction, text: str | None = None) -> list[str]:
    """Empty list = the action is executable (and, if `text` is given, grounded in it)."""
    problems = executable_problems(action)
    if text is not None:
        problems += grounding_problems(action, text)
    return problems
