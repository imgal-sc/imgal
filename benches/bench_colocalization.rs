use divan;
use ndarray::{Array2, Array3};
use rand::Rng;

use imgal::colocalization::{saca_2d, saca_3d};
use imgal::threshold::global::otsu_value;

fn main() {
    divan::main();
}

// Benchmark the colocalization namespace.
#[divan::bench(args = [64, 128, 256])]
fn bench_saca_2d_parallel(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let a = Array2::from_shape_fn((size, size), |_| rng.random::<f64>());
            let b = Array2::from_shape_fn((size, size), |_| rng.random::<f64>());
            let a_ths = otsu_value(&a, None).unwrap();
            let b_ths = otsu_value(&b, None).unwrap();

            (a, b, a_ths, b_ths)
        })
        .bench_values(|(a, b, ta, tb)| {
            let _res = saca_2d(&a, &b, ta, tb, true).unwrap();
        });
}

#[divan::bench(args = [64, 128, 256])]
fn bench_saca_2d_sequential(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let a = Array2::from_shape_fn((size, size), |_| rng.random::<f64>());
            let b = Array2::from_shape_fn((size, size), |_| rng.random::<f64>());
            let a_ths = otsu_value(&a, None).unwrap();
            let b_ths = otsu_value(&b, None).unwrap();

            (a, b, a_ths, b_ths)
        })
        .bench_values(|(a, b, ta, tb)| {
            let _res = saca_2d(&a, &b, ta, tb, false).unwrap();
        });
}

#[divan::bench(args = [64, 128, 256])]
fn bench_saca_3d_parallel(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let a = Array3::from_shape_fn((size, size, size), |_| rng.random::<f64>());
            let b = Array3::from_shape_fn((size, size, size), |_| rng.random::<f64>());
            let a_ths = otsu_value(&a, None).unwrap();
            let b_ths = otsu_value(&b, None).unwrap();

            (a, b, a_ths, b_ths)
        })
        .bench_values(|(a, b, ta, tb)| {
            let _res = saca_3d(&a, &b, ta, tb, true).unwrap();
        });
}

#[divan::bench(args = [64, 128, 256])]
fn bench_saca_3d_sequential(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let mut rng = rand::rng();
            let a = Array3::from_shape_fn((size, size, size), |_| rng.random::<f64>());
            let b = Array3::from_shape_fn((size, size, size), |_| rng.random::<f64>());
            let a_ths = otsu_value(&a, None).unwrap();
            let b_ths = otsu_value(&b, None).unwrap();

            (a, b, a_ths, b_ths)
        })
        .bench_values(|(a, b, ta, tb)| {
            let _res = saca_3d(&a, &b, ta, tb, false).unwrap();
        });
}
