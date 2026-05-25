"""
BFCL → OpenAI function-calling schema conversion.

BFCL function docs use two type names that are NOT valid JSON Schema:
  "dict"  → must become "object"  (OpenAI / JSON Schema standard)
  "float" → must become "number"  (JSON Schema has no "float" type)

Both conversions must happen *recursively* — they appear at the top-level
``parameters`` object AND inside nested property schemas (e.g.
``edit_ticket.updates`` is itself of type "dict").

The BFCL docs also include a "response" key that describes the tool's return
value.  OpenAI's tool schema format has no such field; it must be omitted.
"""
import copy
import json
from pathlib import Path

# Map of BFCL-specific type names → JSON Schema equivalents
_TYPE_MAP: dict[str, str] = {
    "dict": "object",
    "float": "number",
}


def _normalize_schema(schema: object) -> object:
    """
    Recursively walk a schema dict (or list) and replace any non-standard
    ``"type"`` values with their JSON Schema equivalents.
    """
    if isinstance(schema, dict):
        result: dict = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                result[key] = _TYPE_MAP.get(value, value)
            else:
                result[key] = _normalize_schema(value)
        return result
    if isinstance(schema, list):
        return [_normalize_schema(item) for item in schema]
    return schema


def bfcl_func_to_openai_tool(bfcl_func: dict) -> dict:
    """
    Convert one BFCL function-doc entry to the OpenAI tool format.

    Handles:
    - "dict" → "object"  (top-level AND nested)
    - "float" → "number" (top-level AND nested)
    - Strips the BFCL-only "response" field (not part of OpenAI's schema)

    Args:
        bfcl_func: A single parsed line from a BFCL *_func_doc/*.json file.

    Returns:
        An OpenAI-compatible tool dict:
        ``{"type": "function", "function": {"name": ..., "description": ...,
           "parameters": {...}}}``
    """
    params = _normalize_schema(copy.deepcopy(bfcl_func.get("parameters", {})))
    return {
        "type": "function",
        "function": {
            "name": bfcl_func["name"],
            "description": bfcl_func["description"],
            "parameters": params,
        },
    }


# Mapping: simulator class name → JSONL doc filename (in multi_turn_func_doc/)
CLASS_DOC_FILE: dict[str, str] = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def load_tools_for_classes(
    involved_classes: list[str],
    docs_dir: Path,
) -> list[dict]:
    """
    Load and convert all tool schemas for the given simulator class names.

    Args:
        involved_classes: e.g. ``["TradingBot", "MathAPI"]``
        docs_dir:         Path to ``bfcl_eval/data/multi_turn_func_doc/``

    Returns:
        List of OpenAI tool dicts, ready to pass to ``chat.completions.create``.
    """
    tools: list[dict] = []
    for cls in involved_classes:
        doc_path = docs_dir / CLASS_DOC_FILE[cls]
        with open(doc_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tools.append(bfcl_func_to_openai_tool(json.loads(line)))
    return tools
