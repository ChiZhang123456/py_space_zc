from setuptools import setup, find_packages

setup(
    name='py_space_zc',
    version='0.1',
    packages=find_packages(),
    author='Chi Zhang',
    author_email='zhangchi9508@gmail.com',
    description='Space physics analysis tools',
    license="MIT",
    install_requires=["numpy",
                      "matplotlib",
                      "scipy",
                      "pandas",
                      "spacepy",
                      "spiceypy",
                      "pyrfu",
                      "os",
                      ],
)
