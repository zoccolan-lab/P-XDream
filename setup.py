"""
This file is used to install the package pxdream.

Run the following command from the root directory of the package to install the package:
    ```pip install -e .```

"""

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pxdream',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add any dependencies your package requires
        *requirements
    ],
    entry_points={
        'console_scripts': [
            # Add any console scripts your package provides
        ],
    },
)