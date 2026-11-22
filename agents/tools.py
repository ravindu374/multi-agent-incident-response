"""Read-only evidence tools. An agent may only call the tools listed on its own Agent Card.

Where the answers come from is pluggable:
  RecordedEvidence  the evidence bundle stored with each incident (synthetic data, or replayed
                    sandbox captures). Tool calls are dictionary look-ups.
  (later) live tools querying the running sandbox, with the same names and return shapes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from controller.cards import DOMAIN_TOOLS
from controller.evidence import EvidenceBundle, load_evidence_bundles

TOOL_SECTIONS = {tool: path for tools in DOMAIN_TOOLS.values() for tool, path in tools.items()}


class ToolNotAllowed(PermissionError):
    pass


class EvidenceUnavailable(LookupError):
    """The tool ran but returned nothing (evidence missing for this incident)."""


class EvidenceSource(Protocol):
    def call(self, incident_id: str, tool: str) -> Any: ...


class RecordedEvidence:
    def __init__(self, bundles: dict[str, EvidenceBundle]):
        self.bundles = bundles

    @classmethod
    def from_incidents_file(cls, path: Path) -> RecordedEvidence:
        return cls(load_evidence_bundles(path))

    def call(self, incident_id: str, tool: str) -> Any:
        if tool not in TOOL_SECTIONS:
            raise KeyError(f"unknown tool {tool!r}")
        bundle = self.bundles.get(incident_id)
        if bundle is None:
            raise EvidenceUnavailable(f"no evidence recorded for {incident_id}")
        section, field = TOOL_SECTIONS[tool].split(".")
        value = getattr(getattr(bundle, section), field)
        if value in ([], {}, None):
            raise EvidenceUnavailable(f"{tool} returned nothing")
        return value


class Toolbox:
    """The tools one agent is allowed to use, with a log of what it called."""

    def __init__(self, source: EvidenceSource, allowed: list[str], incident_id: str):
        self.source, self.allowed, self.incident_id = source, set(allowed), incident_id
        self.calls: list[str] = []
        self.results: dict[str, Any] = {}  # what the agent saw (used to check its proposal is grounded)

    def call(self, tool: str) -> Any:
        if tool not in self.allowed:
            raise ToolNotAllowed(f"tool {tool!r} is not on this agent's card")
        self.calls.append(tool)
        value = self.source.call(self.incident_id, tool)
        self.results[tool] = value
        return value

    def try_call(self, tool: str) -> Any | None:
        """Call a tool; None if it is not allowed or returned nothing (the reasoner then knows less)."""
        try:
            return self.call(tool)
        except (ToolNotAllowed, EvidenceUnavailable):
            return None
