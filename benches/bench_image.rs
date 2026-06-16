use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::arr2;

use imgal::image::histogram;
use imgal::simulation::blob::gaussian_metaballs;

const SIZE: usize = 1024;
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const THREADS: Option<usize> = Some(0);

fn bench_histogram_par(c: &mut Criterion) {
    let center = [[5.0, SIZE as f64 / 2.0, SIZE as f64 / 2.0]];
    let shape = [10, SIZE, SIZE];
    let data = gaussian_metaballs(
        &arr2(&center),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &shape,
        None,
    )
    .unwrap();
    c.bench_function("histogram (Parallel)", |b| {
        b.iter(|| {
            let _ = histogram(&data, None, THREADS).unwrap();
        })
    });
}

fn bench_histogram_seq(c: &mut Criterion) {
    let center = [[5.0, SIZE as f64 / 2.0, SIZE as f64 / 2.0]];
    let shape = [10, SIZE, SIZE];
    let data = gaussian_metaballs(
        &arr2(&center),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &shape,
        None,
    )
    .unwrap();
    c.bench_function("histogram (Sequential)", |b| {
        b.iter(|| {
            let _ = histogram(&data, None, Some(1)).unwrap();
        })
    });
}

criterion_group!(benches, bench_histogram_par, bench_histogram_seq);
criterion_main!(benches);
