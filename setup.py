from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_include

topo_te_ext = Pybind11Extension(
    "topoconscious.ext._topo_te",
    ["topoconscious/ext/topo_te.cpp"],
    cxx_std=17,
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="topoconscious",
    use_scm_version=True,
    author="Felix Chege Ng'ang'a",
    description="Persistent homology pipeline for neural correlates of consciousness",
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
