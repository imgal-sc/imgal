use criterion::{Criterion, criterion_group, criterion_main};
use imgal::gpu::*;
use ndarray::arr2;

use imgal::simulation::blob::gaussian_metaballs;
use imgal::statistics::{linear_percentile, max, min, min_max, sum, sum_gpu};

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

fn bench_sum_gpu(c: &mut Criterion) {
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
    let data = data.mapv(|v| v as f32);
    init_gpu();
    let input = GpuTensor::new(&data);
    let output = GpuTensor::<f32>::empty_one();
    c.bench_function("sum_gpu", |b| {
        b.iter(|| {
            let _ = sum_gpu(&input, &output);
        });
    });
}

criterion_group!(
    benches,
    bench_linear_percentile,
    bench_max,
    bench_min,
    bench_min_max,
    bench_sum,
    bench_sum_gpu,
);
criterion_main!(benches);
