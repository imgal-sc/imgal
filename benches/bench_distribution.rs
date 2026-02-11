use divan::Bencher;

use imgal::distribution::{inverse_normal_cdf, normalized_gaussian};

const SIGMA: f64 = 3.0;
const RANGE: f64 = 150.0;
const CENTER: f64 = 89.5;

fn main() {
    divan::main();
}

#[divan::bench(args = [0.20, 0.50, 0.99])]
fn bench_inverse_normal_cdf(bencher: Bencher, prob: f64) {
    bencher.bench(|| {
        let _ = inverse_normal_cdf(prob);
    });
}

#[divan::bench(args = [1000, 100_000, 1_000_000])]
fn bench_normalized_gaussian_parallel(bencher: Bencher, bins: usize) {
    bencher.bench(|| {
        let _ = normalized_gaussian(SIGMA, bins, RANGE, CENTER, true);
    });
}

#[divan::bench(args = [1000, 100_000, 1_000_000])]
fn bench_normalized_gaussian_sequential(bencher: Bencher, bins: usize) {
    bencher.bench(|| {
        let _ = normalized_gaussian(SIGMA, bins, RANGE, CENTER, false);
    });
}
