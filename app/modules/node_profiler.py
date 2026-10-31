"""Rank node execution time from history JSON. Read-only: no writes, no HTTP."""

from __future__ import annotations

from collections import defaultdict


def _graph_from_entry(entry: dict) -> dict:
    prompt = entry.get("prompt")
    if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
        return prompt[2]
    if isinstance(prompt, dict):
        return prompt
    return {}


def _class_map(graph: dict) -> dict[str, str]:
    mapping = {}
    for node_id, node in graph.items():
        if isinstance(node, dict) and node.get("class_type"):
            mapping[str(node_id)] = str(node["class_type"])
    return mapping


def timings_from_node_list(items: list) -> list[dict]:
    timings = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if "duration_ms" not in item:
            continue
        timings.append(
            {
                "node_id": str(item.get("node_id")),
                "class_type": str(item.get("class_type") or "Unknown"),
                "duration_ms": float(item["duration_ms"]),
            }
        )
    return timings


def timings_from_messages(messages: list, class_types: dict[str, str]) -> list[dict]:
    """Duration is the gap until the next timestamped execution event."""
    events = []
    for item in messages or []:
        if not isinstance(item, list) or len(item) < 2 or not isinstance(item[1], dict):
            continue
        kind, payload = item[0], item[1]
        if "timestamp" not in payload:
            continue
        if kind == "executing" and payload.get("node"):
            events.append((int(payload["timestamp"]), str(payload["node"])))
        elif kind == "execution_success":
            events.append((int(payload["timestamp"]), None))
    timings = []
    for index, (stamp, node_id) in enumerate(events):
        if node_id is None or index + 1 >= len(events):
            continue
        delta = events[index + 1][0] - stamp
        timings.append(
            {
                "node_id": node_id,
                "class_type": class_types.get(node_id, "Unknown"),
                "duration_ms": float(delta),
            }
        )
    return timings


def collect_timings(history: dict) -> list[dict]:
    if not isinstance(history, dict):
        raise ValueError("history must be an object")
    collected = []
    for entry in history.values():
        if not isinstance(entry, dict):
            continue
        graph = _graph_from_entry(entry)
        classes = _class_map(graph)
        if entry.get("node_timings"):
            collected.extend(timings_from_node_list(entry["node_timings"]))
            continue
        status = entry.get("status") or {}
        collected.extend(timings_from_messages(status.get("messages") or [], classes))
    return collected


def rank_nodes(history: dict) -> dict:
    timings = collect_timings(history)
    by_type: dict[str, dict] = defaultdict(lambda: {"total_ms": 0.0, "count": 0})
    by_node: dict[str, dict] = defaultdict(lambda: {"total_ms": 0.0, "count": 0, "class_type": "Unknown"})
    for timing in timings:
        bucket = by_type[timing["class_type"]]
        bucket["total_ms"] += timing["duration_ms"]
        bucket["count"] += 1
        node = by_node[timing["node_id"]]
        node["total_ms"] += timing["duration_ms"]
        node["count"] += 1
        node["class_type"] = timing["class_type"]
    type_rows = [
        {
            "class_type": name,
            "total_ms": values["total_ms"],
            "count": values["count"],
            "mean_ms": values["total_ms"] / values["count"] if values["count"] else 0,
        }
        for name, values in by_type.items()
    ]
    node_rows = [
        {
            "node_id": node_id,
            "class_type": values["class_type"],
            "total_ms": values["total_ms"],
            "count": values["count"],
            "mean_ms": values["total_ms"] / values["count"] if values["count"] else 0,
        }
        for node_id, values in by_node.items()
    ]
    type_rows.sort(key=lambda row: (-row["total_ms"], row["class_type"]))
    node_rows.sort(key=lambda row: (-row["total_ms"], row["node_id"]))
    return {
        "read_only": True,
        "deleted": [],
        "comfyui_called": False,
        "timings": timings,
        "by_type": type_rows,
        "by_node": node_rows,
        "slowest_type": type_rows[0]["class_type"] if type_rows else None,
        "slowest_node": node_rows[0] if node_rows else None,
    }
