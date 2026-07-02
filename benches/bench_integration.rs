use criterion::{Criterion, criterion_group, criterion_main};

use imgal::distribution::normalized_gaussian;
use imgal::integration::{composite_simpson, midpoint, simpson};

const SIGMA: f64 = 2.0;
const BINS: usize = 512;
const BINS_EVEN_SUB: usize = 511;
const RANGE: f64 = 4.0;
const CENTER: f64 = 2.0;
const THREADS: Option<usize> = Some(0);

fn bench_midpoint(c: &mut Criterion) {
    let data = normalized_gaussian(SIGMA, BINS, RANGE, CENTER, None);
    let mut group = c.benchmark_group("midpoint");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = midpoint(&data, None, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = midpoint(&data, None, Some(1));
        });
    });
    group.finish();
}

fn bench_simpson(c: &mut Criterion) {
    let data = normalized_gaussian(SIGMA, BINS_EVEN_SUB, RANGE, CENTER, None);
    let mut group = c.benchmark_group("simpson");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = simpson(&data, None, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = simpson(&data, None, Some(1));
        });
    });
    group.finish();
}

fn bench_composite_simpson(c: &mut Criterion) {
    let data = normalized_gaussian(SIGMA, BINS, RANGE, CENTER, None);
    let mut group = c.benchmark_group("composite_simpson");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = composite_simpson(&data, None, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = composite_simpson(&data, None, Some(1));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_midpoint,
    bench_simpson,
    bench_composite_simpson,
);
criterion_main!(benches);
