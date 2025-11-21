import numpy as np

from hippocampus_core.conjunctive_place_cells import (
    ConjunctivePlaceCellConfig,
    ConjunctivePlaceCellPopulation,
)


def test_conjunctive_population_shapes():
    config = ConjunctivePlaceCellConfig(
        num_place_cells=5, grid_dim=4, head_direction_dim=3
    )
    population = ConjunctivePlaceCellPopulation(config)
    grid = np.ones(config.grid_dim)
    hd = np.ones(config.head_direction_dim)

    rates = population.compute_rates(grid, hd)
    assert rates.shape == (config.num_place_cells,)
    assert np.all(rates >= 0.0)

