use divan::Bencher;
use imgal::statistics::{linear_percentile, max, min, min_max};
use ndarray::Array3;
use rand::Rng;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_linear_percentile(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| linear_percentile(&data, 0.90, None, None));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_max_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| max(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| min(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_max_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| min_max(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| min(&data, false));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_max_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| max(&data, false));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_max_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| min_max(&data, false));
}
