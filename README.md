# py_space_zc

`py_space_zc` is a Python package for space plasma physics and planetary spacecraft data analysis. It is developed by Chi Zhang for research workflows involving MAVEN, Tianwen-1, Venus Express, EMM, coordinate transformations, velocity distribution functions, magnetic field analysis, plasma moments, and planetary boundary models.

The package is research oriented and currently in active development.

## Main Capabilities

- Time conversion and time series helpers based on `numpy`, `xarray`, and `pyrfu`
- CDF reading utilities for spacecraft data products
- Vector operations, tensor rotations, LMN coordinates, MVA, and field aligned coordinates
- MSO, MSE, spacecraft frame, and planetary coordinate transformations
- MAVEN tools for MAG, SWIA, STATIC, SWEA, SPICE geometry, bow shock, MPB, and crustal field analysis
- Tianwen-1 and Venus Express helper modules
- Velocity distribution function processing, pitch angle distributions, reduced distributions, and plasma moments
- Plotting helpers for time series, spectrograms, skymaps, spacecraft trajectories, and planetary maps
- Ionization, sputtering, and neutral atmosphere helper routines

## Plot Font

Package plotting helpers use Roboto bold as the default font through `pyfonts`:

```python
from pyfonts import load_google_font

font = load_google_font("Roboto", weight="bold")
```

The shared plotting utilities configure Matplotlib with this font automatically, including axis labels, tick labels, titles, legends, colorbar labels, and math text where possible.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/ChiZhang123456/py_space_zc.git
```

The Mars crustal magnetic field tools use `pymagglobal`, which is distributed through the GFZ GitLab PyPI registry. If `pymagglobal` cannot be found from the default PyPI index, install with the GFZ extra index:

```bash
pip install git+https://github.com/ChiZhang123456/py_space_zc.git --extra-index-url https://<GFZ_USER>:<GFZ_TOKEN>@git.gfz-potsdam.de/api/v4/projects/1055/packages/pypi/simple
```

If `pymagglobal` is already installed, the normal GitHub install command is sufficient.

## Dependencies

Core dependencies are declared in `setup.py`. They include:

- `numpy`, `scipy`, `pandas`, `xarray`
- `matplotlib`, `cartopy`, `pyvista`
- `spacepy`, `cdflib`, `spiceypy`, `pyrfu`
- `astropy`, `h5py`, `numba`, `tqdm`, `requests`
- `pyfonts` for loading the package plotting font
- `pyshtools`, `pymagglobal`
- `irfpy` for Venus Express tools

Some scientific Python packages, especially `cartopy`, `spacepy`, and `pyshtools`, may be easier to install with conda on Windows.

## Quick Start

```python
import py_space_zc as pzc

tint = ["2015-01-01T00:00:00", "2015-01-01T01:00:00"]
times = pzc.time_linspace(tint[0], tint[1], 60)

print(pzc.__version__)
```

MAVEN data paths can be initialized with:

```python
from py_space_zc import maven

maven.db_init()
```

Then configure the generated MAVEN data path file for the local machine.

## Package Data

The repository includes small reference files used by several modules, including ionization cross sections, Mars crustal magnetic field coefficients, MAVEN helper files, EMM configuration files, Tianwen-1 configuration files, and sputtering yield tables. These files are included in the Python package during installation.

## Notes

- This package is designed for research use and may change as analysis workflows evolve.
- SPICE based functions require valid kernels and correct local paths.
- Mission data readers assume that local data directories are configured before use.
- The crustal field functions require `pymagglobal` and `pyshtools`.

## Citation

If this package supports a publication, please cite the relevant spacecraft data products, model papers, and analysis methods used in the workflow.

## Author

Chi Zhang

Boston University

Solar wind and Mars interaction, Martian induced magnetosphere, magnetic reconnection, MAVEN and Tianwen-1 data analysis

## License

MIT License. See `LICENSE.txt`.
