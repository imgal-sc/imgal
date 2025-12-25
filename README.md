<div align="center">
  <img src="./docs/assets/imgal_banner.svg" width="350px"/>
  <br />
  <br />

[![crates.io](https://img.shields.io/crates/v/imgal.svg)](https://crates.io/crates/imgal)

</div>

Imgal (**IM**a**G**e **A**lgorithm **L**ibrary) is a fast and open-source
scientific image processing and algorithm library. This library is directly
inspired by [imagej-ops](https://github.com/imagej/imagej-ops/),
[SciJava Ops](https://github.com/scijava/scijava),
[ImgLib2](https://github.com/imglib/imglib2), and the ImageJ2 ecosystem. The `imgal`
library aims to offer users access to fast and well documented image algorithms.
`imgal` is organized as a monorepo with `imgal` as the core library that
contains the algorithm logic while `imgal_java` and `imgal_python` serve
`imgal`'s Java and Python language bindings respectively.

## Usage

### Using `imgal` with Rust

To use `imgal` in your Rust project add it to your crates's dependencies and
import the desired `imgal` namespaces.

```
[dependencies]
imgal = "0.1.0"
```

The example below demonstrates how to create a cube shaped kernel with a
weighted sphere (_i.e._ the neighborhood) of the specified radius and weight
decay rate defined by the falloff radius.

```rust
use imgal::kernel::neighborhood;

fn main() {
  // set radius and weight decay falloff radius
  let radius = 5;
  let falloff = 7.5;

  // create a weighted sphere with given radius and falloff
  let sphere = neighborhood::weighted_sphere(radius, falloff, None);
}
```

### Using `imgal` with Python

You can use `imgal` with Python by using the `imgal_python` PyO3-based Rust
bindings for Python. Pre-compiled releases are available on PyPI as the `pyimgal`
package and can be easily installed with `pip`:

```bash
pip install pyimgal
```

The `pyimgal` package currently supports the following architectures:

| Operating System | Architecture |
| :---             | :---                 |
| Linux            | amd64, aarch64       |
| macOS            | intel, arm64         |
| Windows          | amd64                |

These binaries are compiled for Python `3.9`, `3.10`, `3.11`, `3.12`, and `3.13`.
Alternatively you can build the `imgal_python` package from source with the Rust
toolchain (_i.e._ `rustc` and `cargo`) and the `maturin` Python package. See the
building from source section below for more details.

Once `imgal_python` has been installed in a compatiable Python environment,
`imgal` will be available to import. The example below demonstrates how
to obtain a colocalization z-score (_i.e._ colocalization and anti-colocalization
strength) using the [Spatially Adaptive Colocalization Analysis (SACA)](https://doi.org/10.1109/TIP.2019.2909194)
framework. The two number values after the channels are threshold values for
channels `a` and `b` respectively.

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

You can build `imgal` from the root of the repository with:

```bash
$ cargo build --release
```
> [!NOTE]
>
> `--release` is _necessary_ to compile speed optimized libraries and utilize compiler optimizations.

This will create one Rust static library (`.rlib`) file for `imgal` and two
shared library files for the Java and Python bindings respectively. The file
extension of the shared library is operating system dependent:

| Platform | Extension |
| :---     | :---      |
| Linux    | `.so`     |
| macOS    | `.dylib`  |
| Windows  | `.dll`    |

Additionally, shared libraries will be prefixed with `lib`, making the compiled
`imgal` library filename `libimgal.rlib`. After building `imgal` the three
library files can be found in `target/release`.

| File name | Description |
| :---      | :---        |
| libimgal.rlib | The main Rust static library.
| libimgal.so | Python bindings (using PyO3). |
| libimgal_java.so | Java bindings using the Foreign Function and Memory (FFM) API (targeting Java 22+). |


### Building `imgal_python` from source

To build the `pyimgal` Python package from source, use the `maturin` build tool
(this requires the Rust toolchain). If you're using `uv` to manage your Python
virtual environments (venv) add `maturin` to your environment and run the
`maturin develop --release` command in the `imgal_python` directory of the
[imgal](https://github.com/imgal-sc/imgal) repository with your venv activated:

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

Each function in `imgal` is documented and published [here](https://docs.rs/imgal/).

## License

Imgal is a dual-licensed project with your choice of:

- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))
- MIT License (see [LICENSE-MIT](LICENSE-MIT))
