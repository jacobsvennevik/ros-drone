import numpy as np
import pytest

from hippocampus_core.env import Environment
from hippocampus_core.place_cells import PlaceCellPopulation


@pytest.fixture
def tiny_environment():
    return Environment(width=1.0, height=1.0)


@pytest.fixture
def seeded_population(tiny_environment):
    rng = np.random.default_rng(12345)
    return PlaceCellPopulation(
        environment=tiny_environment,
        num_cells=5,
        sigma=0.12,
        max_rate=18.0,
        rng=rng,
    )


def test_place_cell_rates_are_non_negative(seeded_population):
    rates = seeded_population.get_rates(0.5, 0.5)
    assert np.all(rates >= 0.0)


def test_rate_at_center_is_maximal(seeded_population, tiny_environment):
    centers = seeded_population.get_positions()
    first_center = centers[0]

    bounds = tiny_environment.bounds
    offset = np.array([0.35, -0.3])
    far_point = np.clip(
        first_center + offset,
        [bounds.min_x, bounds.min_y],
        [bounds.max_x, bounds.max_y],
    )
    if np.allclose(far_point, first_center):
        far_point = np.clip(
            first_center - offset,
            [bounds.min_x, bounds.min_y],
            [bounds.max_x, bounds.max_y],
        )

    center_rates = seeded_population.get_rates(*first_center)
    far_rates = seeded_population.get_rates(*far_point)

    assert center_rates[0] >= far_rates[0] - 1e-8

