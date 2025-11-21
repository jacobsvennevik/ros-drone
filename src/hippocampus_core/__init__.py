"""Core hippocampal-inspired simulation components."""

__all__ = ["coactivity", "controllers", "env", "place_cells", "topology", "visualization", "policy"]
__version__ = "2.1.0"
__model_version__ = "hippocampus_core 2.1.0"
"""
Model version identifier for reproducibility tracking.

This version string is written into simulation logs and experimental outputs
to ensure that results can be traced to the exact model revision that generated them.
This is a requirement for reproducibility statements in journals like Nature Neuroscience.

Format: "hippocampus_core <major>.<minor>.<patch>"
- Major version: Significant architectural changes
- Minor version: New features, biological improvements
- Patch version: Bug fixes, numerical stability improvements
"""
