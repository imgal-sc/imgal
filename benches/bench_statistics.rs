use imgal::statistics::{max, min, min_max};
use ndarray::Array3;
use rand::Rng;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_max_parallel(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| max(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_parallel(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| min(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_max_parallel(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| min_max(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_sequential(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| min(&data, false));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_max_sequential(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| max(&data, false));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_max_sequential(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
        })
        .bench_values(|data| min_max(&data, false));
}
