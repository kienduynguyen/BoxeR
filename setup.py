import glob
import os

import torch

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    CppExtension,
    CUDAExtension,
    BuildExtension,
)

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
assert TORCH_VERSION >= (1, 8), "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "detector", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "e2edet", "module", "ops", "src")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        if TORCH_VERSION < (1, 7):
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "e2edet.ops",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="e2edet",
    version="0.1",
    author="VisLab",
    description="Implementation for end-to-end architectures.",
    packages=find_packages(exclude=("tests", "exps", "scripts")),
    python_requires=">=3.6",
    install_requires=[
        "Pillow>=7.1",
        "omegaconf>=2.1",
        "pycocotools",
        "numpy",
        "waymo-open-dataset-tf-2-6-0",
        "matplotlib",
        "numba",
        "scipy",
        "opencv-python",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
