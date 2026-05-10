from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"


setup(
    name="py_space_zc",
    version="2.4.2",
    packages=find_packages(),
    author="Chi Zhang",
    author_email="zhangchi9508@gmail.com",
    description="Space plasma physics analysis tools for planetary spacecraft data",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/ChiZhang123456/py_space_zc",
    license="MIT",
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "py_space_zc": [
            "ionization/*.txt",
            "maven/*.txt",
            "maven/*.sav",
            "maven/*.mat",
            "maven/*.json",
            "maven/*.jpg",
            "maven/*.png",
            "emm/*.json",
            "tianwen_1/*.json",
            "vex/*.json",
            "sputtering/*.mat",
        ],
    },
    install_requires=[
        "astropy",
        "cartopy",
        "cdflib",
        "h5py",
        "irfpy",
        "matplotlib",
        "numba",
        "numpy",
        "pandas",
        "pymagglobal",
        "pyrfu",
        "pyshtools",
        "python-dateutil",
        "pyfonts",
        "pyvista",
        "requests",
        "scipy",
        "spacepy",
        "spiceypy",
        "tqdm",
        "xarray",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
