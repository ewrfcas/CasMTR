#!/usr/bin/env python

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_file = os.path.dirname(__file__)

setup(
    name="score_computation_package",
    ext_modules=[
        CUDAExtension(
            "fast_score_computation",
            [
                "src/score_computation.cpp",
                "src/score_computation_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
