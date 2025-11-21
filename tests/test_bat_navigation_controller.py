import numpy as np

from hippocampus_core.controllers.bat_navigation_controller import (
    BatNavigationController,
    BatNavigationControllerConfig,
)
from hippocampus_core.env import Environment


def test_bat_navigation_controller_runs():
    env = Environment(width=1.0, height=1.0)
    config = BatNavigationControllerConfig(num_place_cells=12, calibration_interval=5)
    controller = BatNavigationController(environment=env, config=config)

    obs = np.array([0.5, 0.5, 0.0])
    action = controller.step(obs, dt=0.1)

    assert action.shape == (2,)
    graph = controller.get_graph()
    assert graph.num_nodes() == config.num_place_cells

