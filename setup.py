from setuptools import setup, find_packages

setup(
    name='py_space_zc',
    version='1.0',
    packages=find_packages(),
    author='Chi Zhang',
    author_email='zhangchi9508@gmail.com',
    description='Space physics analysis tools',
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
                      'pymagglobal @ https://public:5mz_iyigu-WE3HySBH1J@git.gfz-potsdam.de/api/v4/projects/1055/packages/pypi/files/pymagglobal-0.1.0.tar.gz',
                      "xarray",],
)
