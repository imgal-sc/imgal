use divan::Bencher;

use imgal::image::histogram;
use imgal::simulation::gradient::linear_gradient_2d;

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
