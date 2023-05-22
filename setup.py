# setup.py
from setuptools import setup

setup(
    name='roar_py',
    version='0.0.1',
    packages=['roar_py_interface', 'roar_py_carla_implementation', 'roar_py_remote', 'roar_py_client'],
    python_requires='>=3.8',
)