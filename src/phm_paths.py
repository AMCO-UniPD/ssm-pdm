"""Shared output-path resolution for PHM training and post-processing."""

import os
from pathlib import Path


def results_root(default):
    """Use the campaign's recorded result directory, or legacy script-local paths."""
    return Path(os.environ.get("PHM_RESULTS_DIR", default)).expanduser().resolve()


def campaign_results_root(directory, manifest, default):
    """Resolve post-processing paths for current and relocated saved campaigns."""
    directory = Path(directory).expanduser().resolve()
    if os.environ.get("PHM_RESULTS_DIR"):
        return results_root(default)
    recorded_environment = manifest.get("environment", {})
    recorded = manifest.get("results_root") or recorded_environment.get("PHM_RESULTS_DIR")
    if recorded:
        path = Path(recorded).expanduser()
        previous = recorded_environment.get("PHM_MONITOR_DIR")
        if previous:
            try:
                path = directory / path.relative_to(Path(previous).expanduser())
            except ValueError:
                pass  # A separately recorded results directory remains authoritative.
        return path.resolve()
    if (directory / "results").is_dir():
        return directory / "results"
    return Path(default).expanduser().resolve()
