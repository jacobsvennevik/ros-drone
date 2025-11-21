"""Logging utilities for model version tracking.

This module provides utilities for logging model version information
to ensure reproducibility in experimental outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from hippocampus_core import __model_version__, __version__


def log_model_metadata(
    output_path: Path | str,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log model version and metadata to a JSON file.
    
    This ensures that experimental results can be traced to the exact
    model revision that generated them (required for reproducibility in
    journals like Nature Neuroscience).
    
    Parameters
    ----------
    output_path:
        Path to output JSON file (will be created or appended to)
    additional_metadata:
        Optional additional metadata to include (e.g., parameters, timestamps)
    
    Example
    -------
    >>> from hippocampus_core.utils.logging import log_model_metadata
    >>> log_model_metadata("results/experiment_1/metadata.json", {
    ...     "experiment_name": "grid_spacing_validation",
    ...     "parameters": {"grid_size": (20, 20), "tau": 0.05},
    ... })
    """
    metadata = {
        "model_version": __model_version__,
        "package_version": __version__,
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to existing metadata if file exists
    if output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
            # If existing is a list, append; otherwise wrap in list
            if isinstance(existing, list):
                existing.append(metadata)
                metadata = existing
            else:
                metadata = [existing, metadata]
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_model_version() -> str:
    """Return the current model version string.
    
    Returns
    -------
    str
        Model version string (e.g., "hippocampus_core 2.1.0")
    """
    return __model_version__


def print_model_info() -> None:
    """Print model version information to stdout.
    
    Useful for logging at the start of simulations or notebooks.
    """
    print(f"Model Version: {__model_version__}")
    print(f"Package Version: {__version__}")

