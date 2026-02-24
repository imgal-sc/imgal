<div align="center">
  <img src="https://github.com/imgal-sc/imgal/blob/main/docs/assets/png/imgal_banner.png?raw=true" width="350px"/>

[![crates.io](https://img.shields.io/crates/v/imgal)](https://crates.io/crates/imgal)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

Imgal (**IM**a**G**e **A**lgorithm **L**ibrary) is a fast and open-source scientific image processing and algorithm library.
This library is directly inspired by [imagej-ops](https://github.com/imagej/imagej-ops/), [SciJava Ops](https://github.com/scijava/scijava),
[ImgLib2](https://github.com/imglib/imglib2), and the ImageJ2 ecosystem. The imgal library aims to offer users access to fast and **well documented**
image algorithms as a functional programming style library. Imgal is organized as a monorepo with the `imgal` crate as the core Rust library that
contains the algorithm logic while `imgal_c`, `imgal_java` and `imgal_python` serve imgal's C, Java and Python language bindings respectively.

## Usage

### Using imgal with Rust

To use imgal in your Rust project add it to your crate's dependencies and import the desired algorithm namespaces.

```toml
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

## Building from source

Although its not particularly useful on its own, you can build the imgal core Rust library from the root of the
repository with:

```bash
$ cargo build --release
```
> [!NOTE]
>
> `--release` is _necessary_ to compile speed optimized libraries and utilize compiler optimizations.

## Benchmarks

Imgal uses `divan` for benchmarks. You can run all the benchmarks at once with:

```bash
$ cargo bench
```

Or all the benchmarks in a given namespace (*e.g.* statistics) with:

```bash
$ cargo bench -- statistics
```

Or a specific subset of benchmarks (*i.e.* run only the parallel statistics benchmarks):

```bash
$ cargo bench --bench statistics -- parallel
```
 
## Documentation

Each function in `imgal` is documented and published on [docs.rs](https://docs.rs/imgal/).

## License

Imgal is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))
