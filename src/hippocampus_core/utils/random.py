"""Central RNG registry for reproducible random number generation.

This module provides a centralized registry for managing random number generators
across modules, ensuring reproducibility without relying on user discipline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class RNGRegistry:
    """Central registry for managing RNGs across modules.
    
    This registry ensures that each module gets a deterministic RNG based on
    a shared base seed, without requiring manual seed management by users.
    
    Example
    -------
    >>> base_seed = 42
    >>> grid_rng = RNGRegistry.get("grid_cells", seed=base_seed + 2)
    >>> hd_rng = RNGRegistry.get("hd_cells", seed=base_seed + 3)
    >>> # Both RNGs are deterministic and reproducible
    """
    
    _rngs: dict[str, np.random.Generator] = {}
    
    @classmethod
    def get(cls, name: str, seed: Optional[int] = None) -> np.random.Generator:
        """Get or create an RNG for the given module name.
        
        Parameters
        ----------
        name:
            Module name identifier (e.g., "grid_cells", "hd_cells", "place_cells").
        seed:
            Optional seed for the RNG. If provided and RNG doesn't exist yet,
            creates new RNG with this seed. If RNG already exists, returns existing
            RNG regardless of seed (to preserve determinism).
        
        Returns
        -------
        np.random.Generator
            The RNG for this module name. Same RNG is returned on subsequent calls
            with the same name, ensuring reproducibility.
        """
        if name not in cls._rngs:
            if seed is not None:
                cls._rngs[name] = np.random.default_rng(seed)
            else:
                # Create unseeded RNG if no seed provided
                cls._rngs[name] = np.random.default_rng()
        return cls._rngs[name]
    
    @classmethod
    def reset(cls, name: Optional[str] = None) -> None:
        """Reset RNG(s) in the registry.
        
        Parameters
        ----------
        name:
            Optional module name to reset. If None, resets all RNGs.
            This is useful for testing or resetting state between runs.
        """
        if name is None:
            cls._rngs.clear()
        else:
            cls._rngs.pop(name, None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all RNGs from the registry."""
        cls._rngs.clear()

