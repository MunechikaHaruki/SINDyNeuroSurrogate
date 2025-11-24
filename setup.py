# mypy: ignore-errors
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extensions = [
    Extension(
        "neurosurrogate.dataset_utils.traub.traub_simulator",
        sources=[
            "neurosurrogate/dataset_utils/traub/traub_simulator.pyx",
            "neurosurrogate/dataset_utils/traub/traub.c",
        ],
        include_dirs=["./neurosurrogate/dataset_utils/traub", np.get_include()],
    ),
    Extension(
        "neurosurrogate.dataset_utils.hh.hh_simulator",
        sources=[
            "neurosurrogate/dataset_utils/hh/hh_simulator.pyx",
        ],
        include_dirs=["./neurosurrogate/dataset_utils/hh", np.get_include()],
    ),
]


setup(
    ext_modules=cythonize(
        extensions,
        build_dir=os.path.join("build", "cython_c"),
        compiler_directives={"language_level": "3"},
    ),
    packages=find_packages(),
)
