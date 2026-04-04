use divan::Bencher;
use imgal::constants::RNG_SEED;
use imgal::simulation::rng::Pcg;
use imgal::statistics::{linear_percentile, max, min, min_max, sum};
use ndarray::Array3;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_linear_percentile(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| linear_percentile(&data, 0.90, None, None, false));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_max_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| max(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| min(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_max_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| min_max(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_sum_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| sum(&data, true));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| min(&data, false));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_max_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| max(&data, false));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_min_max_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| min_max(&data, false));
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_sum_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut prng = Pcg::new(RNG_SEED);
            Array3::from_shape_fn((size, size, 30), |_| {
                prng.next_u32_range(0..=100u32).unwrap()
            })
        })
        .bench_values(|data| sum(&data, false));
}
