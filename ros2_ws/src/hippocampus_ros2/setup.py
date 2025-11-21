import os
from glob import glob

from setuptools import setup

package_name = "hippocampus_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "scripts"),
            glob("scripts/*.sh"),
        ),
        (
            os.path.join("share", package_name, "system_tests", "launch"),
            glob("system_tests/launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "system_tests", "worlds"),
            glob("system_tests/worlds/*.world"),
        ),
        (
            os.path.join("share", package_name, "system_tests", "scripts"),
            glob("system_tests/scripts/*.py"),
        ),
        (
            os.path.join("share", package_name, "system_tests"),
            ["system_tests/README.md"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="ROS 2 nodes bridging hippocampus_core controllers",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "snn_brain_node = hippocampus_ros2.nodes.brain_node:main",
            "policy_node = hippocampus_ros2.nodes.policy_node:main",
            "mission_publisher = hippocampus_ros2.nodes.mission_publisher:main",
        ],
    },
)
