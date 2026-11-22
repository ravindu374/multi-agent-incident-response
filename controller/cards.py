"""A2A Agent Cards, plus the routing metadata this project adds through an A2A extension.

The card shape follows the A2A v1.0 specification: camelCase JSON, served at
`/.well-known/agent-card.json`, with `supportedInterfaces` (url + protocolBinding +
protocolVersion), `capabilities` (incl. `extensions`), default modes and `skills`.
Checked against the official a2a-sdk 1.3.1 types by controller.wire.to_sdk_card (strict parse +
REQUIRED fields); tests/test_a2a_server.py runs that check on every card in the pool.

A card is *static*: what an agent is and can do. Live state (up/down, load, failures) belongs
to the registry, never to the card.
"""
from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from . import DOMAINS
from .plan import INTENTS

ROUTING_EXTENSION_URI = "urn:a2a-oncall:routing-profile:v1"
WELL_KNOWN_CARD_PATH = "/.well-known/agent-card.json"
A2A_PROTOCOL_VERSION = "1.0"

# Each expert's read-only tools, mapped to the evidence-bundle section they return.
DOMAIN_TOOLS: dict[str, dict[str, str]] = {
    "network": {"probe_paths": "network.probes"},
    "data": {"active_queries": "data.active_queries", "locks": "data.locks", "dbt_failures": "data.last_dbt_failures",
             "connections_by_app": "data.connections_by_app", "source_sample": "data.source_sample_fields"},
    "infra": {"container_stats": "infra.containers", "top_processes": "infra.top_processes",
              "disk_usage": "infra.disks", "recent_changes": "infra.recent_changes"},
    "security": {"db_client_origins": "security.db_client_origins",
                 "failed_logins": "security.failed_logins_by_ip"},
}
SHARED_TOOLS = ("incident_logs", "job_runs")  # every agent can read the incident itself

# Which intents a domain's agents may propose (enforced again by the policy agent).
DOMAIN_INTENTS: dict[str, tuple[str, ...]] = {
    "network": ("reroute_connection", "restore_config"),
    "data": ("map_schema_fields", "quarantine_records", "stop_session", "restore_config"),
    "infra": ("stop_process", "resize_compute", "free_storage", "restart_component", "restore_config"),
    "security": ("rotate_credential", "block_client", "stop_session"),
}


class A2AModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AgentInterface(A2AModel):
    url: str
    protocol_binding: str = "JSONRPC"  # "JSONRPC" | "GRPC" | "HTTP+JSON"
    protocol_version: str = A2A_PROTOCOL_VERSION


class AgentProvider(A2AModel):
    organization: str
    url: str  # REQUIRED by A2A v1.0 whenever a provider is given


class AgentExtension(A2AModel):
    uri: str
    description: str | None = None
    required: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class AgentCapabilities(A2AModel):
    streaming: bool = False
    push_notifications: bool = False
    extensions: list[AgentExtension] = Field(default_factory=list)


class AgentSkill(A2AModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class RoutingProfile(A2AModel):
    """This project's extension params: what the router needs to know about an agent."""

    role: Literal["specialist", "generalist", "policy"]
    domains: list[str] = Field(default_factory=list)
    tier: Literal["fast", "strong"] = "fast"
    model: str
    cost_units: float  # relative cost of one task (fast specialist = 1)
    expected_latency_s: float
    max_concurrency: int = 1
    tools: list[str] = Field(default_factory=list)
    allowed_intents: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> RoutingProfile:
        if not set(self.domains) <= set(DOMAINS):
            raise ValueError(f"unknown domain(s) in {self.domains}")
        if self.role != "policy" and not self.domains:
            raise ValueError("specialists and generalists must cover at least one domain")
        unknown = set(self.allowed_intents) - set(INTENTS)
        if unknown:
            raise ValueError(f"unknown intent(s) {sorted(unknown)}")
        return self


class AgentCard(A2AModel):
    name: str
    description: str
    supported_interfaces: list[AgentInterface]
    version: str
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    default_input_modes: list[str] = Field(default_factory=lambda: ["application/json", "text/plain"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["application/json"])
    skills: list[AgentSkill] = Field(default_factory=list)
    provider: AgentProvider | None = None
    documentation_url: str | None = None

    def routing_profile(self) -> RoutingProfile:
        for ext in self.capabilities.extensions:
            if ext.uri == ROUTING_EXTENSION_URI:
                return RoutingProfile.model_validate(ext.params)
        raise ValueError(f"agent {self.name!r} has no {ROUTING_EXTENSION_URI} extension")

    def skills_text(self) -> str:
        """All human-readable capability text (for matching incidents to cards in Block 4)."""
        parts = [self.description]
        for s in self.skills:
            parts += [s.name, s.description, " ".join(s.tags), " ".join(s.examples)]
        return "\n".join(parts)

    def to_json(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)


def fetch_card(base_url: str, client: httpx.Client | None = None, timeout: float = 5.0) -> AgentCard:
    """A2A discovery: GET {base_url}/.well-known/agent-card.json."""
    url = base_url.rstrip("/") + WELL_KNOWN_CARD_PATH
    if client is None:
        resp = httpx.get(url, timeout=timeout)
    else:
        resp = client.get(url, timeout=timeout)
    resp.raise_for_status()
    return AgentCard.model_validate(resp.json())
