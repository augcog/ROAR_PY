# setup.py
import os
from setuptools import setup, find_packages

def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name='roar_py_carla',
     version='0.1.0',
    description="ROAR_PY carla implementation",
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
    keywords=["reinforcement learning", "gymnasium", "robotics", "carla", "carla-simulator"],
    license="MIT",
    install_requires=read_requirements_file("requirements.txt"),
    packages=['roar_py_carla'], # find_packages(),
    python_requires='>=3.8',
)