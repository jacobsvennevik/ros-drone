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


def test_explicit_centers_are_respected(tiny_environment):
    centers = np.array(
        [
            [0.2, 0.2],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.8, 0.8],
        ],
        dtype=float,
    )

    population = PlaceCellPopulation(
        environment=tiny_environment,
        num_cells=len(centers),
        sigma=0.1,
        max_rate=15.0,
        centers=centers,
    )

    assert np.allclose(population.get_positions(), centers)


def test_orientation_requires_theta(tiny_environment):
    population = PlaceCellPopulation(
        environment=tiny_environment,
        num_cells=3,
        sigma=0.1,
        orientation_kappa=1.5,
        orientation_preferences=np.zeros(3),
    )
    with pytest.raises(ValueError, match="theta must be provided"):
        population.get_rates(0.5, 0.5)


def test_orientation_modulates_rates(tiny_environment):
    population = PlaceCellPopulation(
        environment=tiny_environment,
        num_cells=3,
        sigma=0.1,
        orientation_kappa=2.0,
        orientation_preferences=np.zeros(3),
    )

    aligned = population.get_rates(0.5, 0.5, theta=0.0)
    opposite = population.get_rates(0.5, 0.5, theta=np.pi)

    assert np.all(aligned > opposite)


def test_altitude_requires_z(tiny_environment):
    centers = np.tile(np.array([[0.5, 0.5, 0.25]]), (2, 1))
    population = PlaceCellPopulation(
        environment=tiny_environment,
        num_cells=2,
        sigma=0.1,
        track_altitude=True,
        centers=centers,
    )

    with pytest.raises(ValueError, match="z must be provided"):
        population.get_rates(0.5, 0.5)

    rates = population.get_rates(0.5, 0.5, z=0.25)
    assert rates.shape == (2,)


def test_vector_observation_support(tiny_environment):
    population = PlaceCellPopulation(
        environment=tiny_environment,
        num_cells=4,
        sigma=0.1,
        track_altitude=True,
        orientation_kappa=1.0,
    )

    obs = np.array([0.4, 0.6, 0.2, np.pi / 4])
    rates_from_vector = population.get_rates(obs)
    rates_manual = population.get_rates(0.4, 0.6, z=0.2, theta=np.pi / 4)

    assert np.allclose(rates_from_vector, rates_manual)

