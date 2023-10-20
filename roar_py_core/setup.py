# setup.py
import os
from setuptools import setup, find_packages

def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name='roar_py_core',
    version='0.1.1',
    description="ROAR_PY interface definitions and streaming capabilities library",
    url="https://github.com/augcog/ROAR_PY",
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    author="Yunhao Cao",
    keywords=["reinforcement learning", "gymnasium", "robotics", "rpc"],
    license="MIT",
    install_requires=[
        "gymnasium>=0.28.1",
        "numpy",
        "asyncio",
        "pyserde[all]>=0.10.4",
        "Pillow>=9.5.0",
        "transforms3d>=0.4.1",
        "websockets>=11.0.1",
        "numba>=0.55.0",
        "shapely>=2.0.0"
    ],
    tests_require=[
        "pytest>=7.3.1",
        "pytest-asyncio>=0.21.0"
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
)