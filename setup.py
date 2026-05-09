from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import pybind11

topo_te_ext = Extension(
    "topoconscious.ext._topo_te",
    sources=["topoconscious/ext/topo_te.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="topoconscious",
    version="0.1.0",
    author="TopoConscious Team",
    description="Topological pipeline for consciousness detection from fMRI",
    packages=find_packages(),
    ext_modules=[topo_te_ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "topoconscious=topoconscious.pipeline:cli_entry",
        ]
    },
)
