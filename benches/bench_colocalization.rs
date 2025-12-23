use divan;
use imgal::colocalization::saca_2d;
use ndarray::Array2;
use rand::Rng;

fn main() {
    divan::main();
}

// Benchmark SACA 2D across sizes. We generate input once per size and reuse in iterations.
#[divan::bench(args = [64, 128, 256])]
fn bench_saca_2d(bencher: divan::Bencher, size: usize) {
    // Pre-generate input arrays outside timing region.
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let a = Array2::from_shape_fn((size, size), |_| rng.random::<f64>());
            let b = Array2::from_shape_fn((size, size), |_| rng.random::<f64>());
            // Simple fixed thresholds for now; could be quantiles.
            (a, b, 0.1_f64, 0.1_f64)
        })
        .bench_values(|(a, b, ta, tb)| {
            // Call the function; unwrap result so work is executed.
            let _res = saca_2d(a.view(), b.view(), ta, tb).unwrap();
        });
}
