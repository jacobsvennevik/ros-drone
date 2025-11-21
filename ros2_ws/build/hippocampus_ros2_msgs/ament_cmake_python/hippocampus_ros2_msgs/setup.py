from setuptools import find_packages
from setuptools import setup

setup(
    name='hippocampus_ros2_msgs',
    version='0.1.0',
    packages=find_packages(
        include=('hippocampus_ros2_msgs', 'hippocampus_ros2_msgs.*')),
)
