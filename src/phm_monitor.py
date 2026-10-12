"""Optional persistent PHM campaign telemetry (one sequential pipeline writer)."""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def emit(kind, **values):
    directory = os.environ.get("PHM_MONITOR_DIR")
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    if kind == "started":
        for key in ("exit_code", "error", "fold", "tau", "epoch"):
            state.pop(key, None)
    if kind == "setting_started":
        for key in ("fold", "tau", "epoch"):
            state.pop(key, None)
    context = {key: state[key] for key in ("fold", "tau", "setting", "ratio") if key in state}
    event = safe({**context, "time": datetime.now(timezone.utc).isoformat(),
                  "kind": kind, **values})
    events_path = root / "events.jsonl"
    # Separate a partial final record left by a crash from the next attempt.
    if events_path.exists() and events_path.stat().st_size:
        with events_path.open("rb") as stream:
            stream.seek(-1, 2)
            needs_newline = stream.read(1) != b"\n"
        if needs_newline:
            with events_path.open("a") as stream:
                stream.write("\n")
    with events_path.open("a") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
    state.update(event)
    if kind in ("fold_started", "quantile_started", "stage"):
        state.pop("epoch", None)
    if kind in ("completed", "failed", "interrupted"):
        state["status"] = kind
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False))
    temporary.replace(path)


def table_payload(frame):
    return {"columns": list(frame.columns), "index": list(frame.index),
            "values": frame.values.tolist()}


def best_fold(events):
    latest = {(event.get("setting"), event["fold"]): event
              for event in events if event["kind"] == "fold_completed"}
    folds = [event for event in latest.values() if event.get("validation_loss") is not None]
    if not folds:
        raise ValueError("No completed folds with finite validation loss")
    return min(folds, key=lambda event: (event["validation_loss"], event["fold"]))


if __name__ == "__main__":
    emit("stage", stage=sys.argv[1], status="running")
