use divan::Bencher;

use imgal::copy::duplicate;
use imgal::simulation::gradient::linear_gradient_3d;

const OFFSET: usize = 5;
const SCALE: f64 = 20.0;

fn main() {
    divan::main();
}

// Benchmark the copy namespace
#[divan::bench(args = [256, 512, 1024])]
fn bench_duplicate_parallel(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let data = linear_gradient_3d(OFFSET, SCALE, (size, size, 15));
            data
        })
        .bench_values(|d| {
            let _dup = duplicate(&d, true);
        });
}

#[divan::bench(args = [256, 512, 1024])]
fn bench_duplicate_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let data = linear_gradient_3d(OFFSET, SCALE, (size, size, 15));
            data
        })
        .bench_values(|d| {
            let _dup = duplicate(&d, false);
        });
}
