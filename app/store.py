"""
In-memory match store — holds parsed match data for the session.
Supports multiple innings per match.
"""
from __future__ import annotations
from typing import Dict, Any, Optional

# match_id -> { "innings": { 1: {...}, 2: {...} }, "team1": ..., "team2": ... }
_matches: Dict[str, Dict[str, Any]] = {}


def save_match(match_id: str, innings_number: int, data: Dict[str, Any]) -> None:
    if match_id not in _matches:
        _matches[match_id] = {"innings": {}, "team1": data.get("team1", ""), "team2": data.get("team2", "")}
    _matches[match_id]["innings"][innings_number] = data
    _matches[match_id]["team1"] = data.get("team1", "")
    _matches[match_id]["team2"] = data.get("team2", "")


def get_match(match_id: str, innings_number: int = 1) -> Optional[Dict[str, Any]]:
    m = _matches.get(match_id)
    if not m:
        return None
    return m["innings"].get(innings_number)


def get_match_meta(match_id: str) -> Optional[Dict[str, Any]]:
    m = _matches.get(match_id)
    if not m:
        return None
    return {
        "match_id": match_id,
        "team1": m["team1"],
        "team2": m["team2"],
        "innings_available": sorted(m["innings"].keys()),
    }


def list_matches() -> list[str]:
    return list(_matches.keys())


def delete_match(match_id: str) -> bool:
    if match_id in _matches:
        del _matches[match_id]
        return True
    return False
