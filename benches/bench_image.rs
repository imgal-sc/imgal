use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::arr2;

use imgal::image::{histogram, histogram_bin_midpoint};
use imgal::simulation::blob::gaussian_metaballs;

const SIZE: usize = 1024;
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const THREADS: Option<usize> = Some(0);

fn bench_histogram(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("histogram");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = histogram(&data, None, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = histogram(&data, None, Some(1)).unwrap();
        });
    });
    group.finish();
}

fn bench_histogram_bin_midpoint(c: &mut Criterion) {
    let index = SIZE / 2;
    let min = 0.0;
    let max = 10.0;
    let bins = SIZE;
    c.bench_function("histogram_bin_midpoint", |b| {
        b.iter(|| {
            let _ = histogram_bin_midpoint(
                black_box(index),
                black_box(min),
                black_box(max),
                black_box(bins),
            )
            .unwrap();
        });
    });
}

criterion_group!(benches, bench_histogram, bench_histogram_bin_midpoint);
criterion_main!(benches);
