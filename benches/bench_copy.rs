use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::{Array3, arr2};

use imgal::copy::{copy_into, copy_into_flat, duplicate};
use imgal::simulation::blob::gaussian_metaballs;

const SIZE: usize = 1024;
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const THREADS: Option<usize> = Some(0);

fn bench_copy_into(c: &mut Criterion) {
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
    let mut dup = Array3::<f64>::zeros((10, SIZE, SIZE)).into_dyn();
    let mut group = c.benchmark_group("copy_into");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = copy_into(&data, dup.view_mut(), THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = copy_into(&data, dup.view_mut(), THREADS);
        });
    });
    group.finish();
}

fn bench_copy_into_flat(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("copy_into_flat");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = copy_into_flat(&data, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = copy_into_flat(&data, Some(1));
        });
    });
    group.finish();
}

fn bench_duplicate(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("duplicate");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = duplicate(&data, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = duplicate(&data, Some(1));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_copy_into,
    bench_copy_into_flat,
    bench_duplicate,
);
criterion_main!(benches);
