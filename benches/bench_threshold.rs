use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::arr2;

use imgal::simulation::blob::gaussian_metaballs;
use imgal::threshold::global::{otsu_mask, otsu_value};

const SIZE: usize = 1024;
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const THREADS: Option<usize> = Some(0);

fn bench_otsu_mask(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("otsu_mask");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = otsu_mask(&data, None, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = otsu_mask(&data, None, Some(1)).unwrap();
        });
    });
    group.finish();
}

fn bench_otsu_value(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("otsu_value");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = otsu_value(&data, None, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = otsu_value(&data, None, Some(1)).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_otsu_mask, bench_otsu_value);
criterion_main!(benches);
