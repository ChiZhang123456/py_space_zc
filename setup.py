from setuptools import setup, find_packages

setup(
    name='py_space_zc',
    version='1.0',
    packages=find_packages(),
    author='Chi Zhang',
    author_email='zhangchi9508@gmail.com',
    description='Space physics analysis tools',
    include_package_data=True,
    package_data={
        "py_space_zc": [
            "ionization/*.txt",
            "maven/*.txt",
            "maven/*.sav",
            "sputtering/*.mat",
            ],
    },
    license="MIT",
    include_package_data=True,
    package_data={
        "py_space_zc": [
            "ionization/*.txt",
            "maven/*.txt",
            "maven/*.sav",
            "maven/*.json",
            "emm/*.json",
            "tianwen_1/*.json",
            "sputtering/*.mat",
        ],
    },
    install_requires=["spacepy",
                      "spiceypy",
                      "pyrfu",
                      'pyshtools',
                      'pyvlasiator',
                      "xarray",],
)
