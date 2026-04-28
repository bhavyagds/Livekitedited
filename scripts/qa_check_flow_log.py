"""
Quick QA utility for Elena room logs.

Usage:
    python scripts/qa_check_flow_log.py --log room_*.log
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _extract_payload(line: str) -> dict | None:
    # Expected format: "<ts>Z | <level> | {json...}"
    parts = line.split(" | ", 2)
    if len(parts) < 3:
        return None
    payload = parts[2].strip()
    if not payload.startswith("{"):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def analyze_log(path: Path) -> int:
    if not path.exists():
        print(f"[ERROR] Log not found: {path}")
        return 2

    events = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        payload = _extract_payload(raw)
        if payload:
            events.append(payload)

    if not events:
        print(f"[WARN] No structured events found in {path}")
        return 1

    event_counts = Counter(evt.get("event", "") for evt in events)
    flow_events = [evt for evt in events if evt.get("event") == "SUPPORT_FLOW_STATE"]
    states = [evt.get("current", "") for evt in flow_events]
    state_counts = Counter(states)

    print(f"Log: {path}")
    print(f"Structured events: {len(events)}")
    print("\nTop events:")
    for name, count in event_counts.most_common(12):
        print(f"- {name}: {count}")

    if states:
        print("\nSupport flow states:")
        for state, count in state_counts.items():
            print(f"- {state}: {count}")

        # Simple health checks
        has_order_path = "awaiting_order_number" in state_counts or "checking_order_number" in state_counts
        has_phone_path = "awaiting_phone_number" in state_counts or "awaiting_phone_confirmation" in state_counts
        if has_order_path:
            print("[OK] Order-number flow states observed.")
        else:
            print("[WARN] No order-number flow states observed.")
        if has_phone_path:
            print("[OK] Phone flow states observed.")
        else:
            print("[WARN] No phone flow states observed.")
    else:
        print("\n[WARN] No SUPPORT_FLOW_STATE events found.")

    duplicate_wait = event_counts.get("TOOL_WAIT_ACK_ENFORCED", 0) > 0 and event_counts.get("TOOL_WAIT_ACK_SELECTED", 0) == 0
    if duplicate_wait:
        print("[WARN] Wait-ack enforcement without selection detected; inspect TTS path.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Elena room log flow events")
    parser.add_argument("--log", required=True, help="Path to room log file")
    args = parser.parse_args()
    return analyze_log(Path(args.log))


if __name__ == "__main__":
    raise SystemExit(main())
