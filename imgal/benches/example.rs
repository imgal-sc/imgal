use imgal::statistics::{max_sequential, max, min_max, min_max_sequential};
use ndarray::Array3;
use rand::Rng;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

// Define algorithm trait for comparison
trait MaxAlgorithm {
    fn compute_max(data: &Array3<u32>) -> u32;
    fn compute_min_max(data: &Array3<u32>) -> (u32, u32);
}

struct Sequential;
impl MaxAlgorithm for Sequential {
    fn compute_max(data: &Array3<u32>) -> u32 {
        max_sequential(data.view().into_dyn())
    }
    fn compute_min_max(data: &Array3<u32>) -> (u32, u32) {
        min_max_sequential(data.view().into_dyn())
    }
}

struct Parallel;
impl MaxAlgorithm for Parallel {
    fn compute_max(data: &Array3<u32>) -> u32 {
        max(data)
    }
    fn compute_min_max(data: &Array3<u32>) -> (u32, u32) {
        min_max(data)
    }
}

#[divan::bench(
    args = [100, 500, 1000],
    types = [Sequential, Parallel]
)]
fn bench_max<Algorithm: MaxAlgorithm>(bencher: divan::Bencher, size: usize) {
    bencher.with_inputs(|| {
        let mut rng = rand::rng();
        Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
    }).bench_values(|data| {
        Algorithm::compute_max(&data)
    });
}

#[divan::bench(
    args = [100, 500, 1000],
    types = [Sequential, Parallel]
)]
fn bench_min_max<Algorithm: MaxAlgorithm>(bencher: divan::Bencher, size: usize) {
    bencher.with_inputs(|| {
        let mut rng = rand::rng();
        Array3::from_shape_fn((size, size, 30), |_| rng.random_range(..=100u32))
    }).bench_values(|data| {
        Algorithm::compute_min_max(&data)
    });
}