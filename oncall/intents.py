"""How the lab executes the controller's intents: each intent maps onto sandbox actions.

On Microsoft Fabric the same intents would map onto Fabric operations instead
(see docs/HANDOVER.md). The ground truth's `valid_fixes` are sandbox actions, so this
table is also how a plan is graded against the known correct fix.
"""
from __future__ import annotations

INTENT_ACTIONS: dict[str, tuple[str, ...]] = {
    "reroute_connection": ("switch_source_endpoint", "switch_warehouse_endpoint"),
    "restore_config": ("rollback_service_config",),
    "map_schema_fields": ("set_field_mapping",),
    "quarantine_records": ("quarantine_rows",),
    "stop_session": ("terminate_backend", "terminate_sessions"),
    "stop_process": ("kill_container_process",),
    "resize_compute": ("set_memory_limit",),
    "free_storage": ("purge_spool",),
    "restart_component": ("restart_service",),
    "rotate_credential": ("rotate_db_credential",),
    "block_client": ("block_ip",),
}


def intent_of(action: str) -> str:
    for intent, actions in INTENT_ACTIONS.items():
        if action in actions:
            return intent
    raise KeyError(f"no intent covers sandbox action {action!r}")
