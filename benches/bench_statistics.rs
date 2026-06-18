use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::arr2;

use imgal::simulation::blob::gaussian_metaballs;
use imgal::statistics::{linear_percentile, max, min, min_max, sum};

const SIZE: usize = 1024;
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const THREADS: Option<usize> = Some(0);

fn bench_linear_percentile(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("linear_percentile");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = linear_percentile(&data, 0.9, None, None, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = linear_percentile(&data, 0.9, None, None, Some(1)).unwrap();
        });
    });
    group.finish();
}

fn bench_max(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("max");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = max(&data, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = max(&data, Some(1)).unwrap();
        });
    });
    group.finish();
}

fn bench_min(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("min");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = min(&data, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = min(&data, Some(1)).unwrap();
        });
    });
    group.finish();
}

fn bench_min_max(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("min_max");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = min_max(&data, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = min_max(&data, Some(1)).unwrap();
        });
    });
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("sum");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = sum(&data, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = sum(&data, Some(1));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_percentile,
    bench_max,
    bench_min,
    bench_min_max,
    bench_sum
);
criterion_main!(benches);
