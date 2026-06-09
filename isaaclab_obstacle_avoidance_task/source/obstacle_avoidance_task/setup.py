"""obstacle_avoidance_task 扩展包安装脚本。"""

from __future__ import annotations

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="obstacle_avoidance_task",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=["psutil"],
    license="Apache-2.0",
    include_package_data=True,
    package_data={
        "obstacle_avoidance_task": [
            "assets/car_4ws_4wd/package.xml",
            "assets/car_4ws_4wd/urdf/*.urdf",
            "assets/car_4ws_4wd/meshes/*.STL",
        ],
    },
    python_requires=">=3.10",
    zip_safe=False,
)
