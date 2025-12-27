<div align="center">
  <img src="https://github.com/imgal-sc/imgal/blob/main/docs/assets/png/imgal_banner.png?raw=true" width="350px"/>

[![crates.io](https://img.shields.io/crates/v/imgal)](https://crates.io/crates/imgal)
[![pypi](https://img.shields.io/pypi/v/pyimgal)](https://pypi.org/project/pyimgal)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

Imgal (**IM**a**G**e **A**lgorithm **L**ibrary) is a fast and open-source scientific image processing and algorithm library.
This library is directly inspired by [imagej-ops](https://github.com/imagej/imagej-ops/), [SciJava Ops](https://github.com/scijava/scijava),
[ImgLib2](https://github.com/imglib/imglib2), and the ImageJ2 ecosystem. The imgal library aims to offer users access to fast and **well documented**
image algorithms as a functional programming style library. Imgal is organized as a monorepo with the `imgal` crate as the core Rust library that
contains the algorithm logic while `imgal_c`, `imgal_java` and `imgal_python` serve imgal's C, Java and Python language bindings respectively.

## Usage

### Using imgal with Rust

To use imgal in your Rust project add it to your crates's dependencies and import the desired algorithm namespaces.

```
[dependencies]
imgal = "0.2.0"
```

The example below demonstrates how to create a 3D linear gradient image (with variable offset, scale and size) and perform simple
image statistics and thresholding:

```rust
use imgal::statistics::{min_max, sum};
use imgal::simulation::gradient;
use imgal::threshold::otsu_value;

fn main() {
    // create 3D linear gradient data
    let offset = 5;
    let scale = 20.0;
    let shape: (usize, usize, usize) = (50, 50, 50);
    let data = gradient::linear_gradient_3d(offset, scale, shape);

    // calculate the Otsu threshold value with an image histogram of 256 bins
    let threshold = otsu_value(&data, Some(256));

    // print image statistics and Otsu threshold
    println!("[INFO] min/max: {:?}", min_max(&data));
    println!("[INFO] sum: {}", sum(&data));
    println!("[INFO] otsu threshold: {}", threshold);
}
```

Running this example with `cargo run` returns the following to the console:

```bash
[INFO] min/max: (0.0, 880.0)
[INFO] sum: 49500000
[INFO] otsu threshold: 417.65625
```

### Using imgal with Python

You can use imgal with Python by using the `imgal_python` crate, a PyO3-based Rust bindings for Python. Pre-compiled releases
are available on PyPI as the `pyimgal` package and can be easily installed with `pip`:

```bash
pip install pyimgal
```

The `pyimgal` package currently supports the following architectures:

| Operating System | Architecture |
| :---             | :---                 |
| Linux            | amd64, aarch64       |
| macOS            | intel, arm64         |
| Windows          | amd64                |

These binaries are compiled for Python `3.9`, `3.10`, `3.11`, `3.12`, and `3.13`. Alternatively you can build the `imgal_python` package from source
with the Rust toolchain (_i.e._ `rustc` and `cargo`) and the `maturin` Python package. See the building from source section below for more details.

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
## Building from source

Although its not particularly useful on its own, you can build the imgal core Rust library from the root of the
repository with:

```bash
$ cargo build --release
```
> [!NOTE]
>
> `--release` is _necessary_ to compile speed optimized libraries and utilize compiler optimizations.

This will compile the entier workspace including the `imgal`, `imgal_c`, `imgal_java` and `imgal_python` crates.

### Building `imgal_python` from source

To build the `pyimgal` Python package from source, use the `maturin` build tool (this requires the Rust toolchain). If you're using `uv`
to manage your Python virtual environments (venv) add `maturin` to your environment and run the `maturin develop --release` command in the
`imgal_python` directory of the [imgal](https://github.com/imgal-sc/imgal) repository with your venv activated:

```bash
$ source ~/path/to/myenv/.venv/bin/activate
$ (myenv) cd imgal_python
$ maturin develop --release
```

Alternatively if you're using `conda` or `mamba` you can do the following:

```bash
$ cd imgal_python
$ mamba activate myenv
(myenv) $ mamba install maturin
...
(myenv) $ maturin develop --release
```

This will install `pyimgal` in the currently active Python environment.

## Documentation

Each function in `imgal` is documented and published on [docs.rs](https://docs.rs/imgal/).

## License

Imgal is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))
