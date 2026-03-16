import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


MEMORY_FILE = "memory.jsonl"


def append_episode(episode: Dict[str, Any], path: str = MEMORY_FILE) -> None:
    """
    Append one run/episode to a JSONL file.
    Each line is a standalone JSON object.
    """
    record = dict(episode)
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_recent_episodes(limit: int = 5, path: str = MEMORY_FILE) -> List[Dict[str, Any]]:
    """
    Load the most recent N episodes from memory.jsonl.
    """
    if not os.path.exists(path):
        return []

    episodes: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines safely
                continue

    return episodes[-limit:]


def format_episodes_for_prompt(episodes: List[Dict[str, Any]]) -> str:
    """
    Convert recent episodes into a compact text block for LLM prompts.
    """
    if not episodes:
        return "No prior episodes."

    lines: List[str] = []
    for ep in episodes:
        ticker = ep.get("ticker", "UNKNOWN")
        final_value = ep.get("final_consensus", ep.get("result", {}).get("final_consensus", "N/A"))
        regime = ep.get("inputs", {}).get("macro", {}).get("regime", "N/A")
        ts = ep.get("timestamp_utc", "N/A")
        lines.append(
            f"- time={ts}, ticker={ticker}, regime={regime}, final_consensus={final_value}"
        )

    return "\n".join(lines)