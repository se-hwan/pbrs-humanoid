from setuptools import find_packages
from distutils.core import setup

setup(
    name='gpuGym',
    version='1.0.1',
    author='Biomimetic Robotics Lab',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib',
                      'pandas',
                      'tensorboard']
)
