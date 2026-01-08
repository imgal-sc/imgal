use divan::Bencher;

use imgal::image::{histogram, percentile_normalize};
use imgal::simulation::gradient::{linear_gradient_2d, linear_gradient_3d};

const OFFSET: usize = 5;
const SCALE: f64 = 20.0;

fn main() {
    divan::main();
}

// Benchmark the image namespace.
#[divan::bench(args = [256, 512, 1024])]
fn bench_histogram_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let data = linear_gradient_2d(OFFSET, SCALE, (size, size));
            data
        })
        .bench_values(|d| {
            let _hist = histogram(&d, Some(256), true);
        });
}

#[divan::bench(args = [256, 512, 1024])]
fn bench_histogram_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let data = linear_gradient_2d(OFFSET, SCALE, (size, size));
            data
        })
        .bench_values(|d| {
            let _hist = histogram(&d, Some(256), false);
        });
}

#[divan::bench(args = [256, 512, 1024])]
fn bench_percentile_normalize_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let data = linear_gradient_3d(OFFSET, SCALE, (size, size, 50));
            data
        })
        .bench_values(|d| {
            let _norm = percentile_normalize(&d, 1.0, 99.8, Some(true), None, true);
        });
}

#[divan::bench(args = [256, 512, 1024])]
fn bench_percentile_normalize_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let data = linear_gradient_3d(OFFSET, SCALE, (size, size, 50));
            data
        })
        .bench_values(|d| {
            let _norm = percentile_normalize(&d, 1.0, 99.8, Some(true), None, false);
        });
}
