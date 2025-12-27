# pyimgal

The pyimgal package provides the Python bindings for the [imgal](https://github.com/imgal-sc/imgal) image algorithm and processing library.
Visit [imgal.org](imgal.org) for more information.

## Installation

### pyimgal from PyPI

You can install the pyimgal package from PyPI with:

```bash
pip install pyimgal
```

The `pyimgal` package supports the following architectures for Python `3.9`, `3.10`, `3.11`, `3.12`, and `3.13`:

| Operating System | Architecture |
| :---             | :---                 |
| Linux            | amd64, aarch64       |
| macOS            | intel, arm64         |
| Windows          | amd64                |

Alternatively, you can install pyimagal from source by building the `imgal_python` repository. See the next section for instructions.

### Build pyimgal from source

To build the `pyimgal` Python package from source, use the `maturin` build tool (this requires the Rust toolchain to build the `imgal` core library).
If you're using `uv` to manage your Python virtual environments (venv) add `maturin` to your environment and run the `maturin develop --release`
command in the `imgal_python` directory of the [imgal](https://github.com/imgal-sc/imgal) repository with your selected venv activated:

```bash
$ source ~/path/to/myenv/.venv/bin/activate
$ (myenv) cd imgal_python
$ maturin develop --release
```

Alternatively, if you're using `conda` or `mamba` you can do the following:

```bash
$ cd imgal_python
$ mamba activate myenv
(myenv) $ mamba install maturin
...
(myenv) $ maturin develop --release
```

This will install pyimgal in the currently active Python environment.

## Usage

### Using pyimgal

Once `imgal_python` has been installed in a compatible Python environment, `imgal` will be available to import. The example below demonstrates how
to obtain a colocalization z-score (_i.e._ colocalization and anti-colocalization strength) using the [Spatially Adaptive Colocalization Analysis (SACA)](https://doi.org/10.1109/TIP.2019.2909194)
framework. The two number values after the channels are threshold values for channels `a` and `b` respectively.

*Note: This example assumes you have 3D data (row, col, ch) to perform colocalization analysis and the `tifffile` package in your environment.*

```python
import imgal.colocalization as coloc
from tifffile import imread

# load some data
image = imread("path/to/data.tif")

# slice channels to perform colocalization analysis
ch_a = image[:, :, 0]
ch_b = image[:, :, 1]

# compute colocalization z-score with SACA 2D
zscore = coloc.saca_2d(ch_a, ch_b, 525, 400)

# apply Bonferroni correction and compute significant pixel mask
mask = coloc.saca_significance_mask(z_score)
```

## Documentation

Each function in `imgal` is documented and published on [docs.rs](https://docs.rs/imgal/).

## License

Imgal is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))
