use criterion::{Criterion, criterion_group, criterion_main};

use imgal::distribution::{inverse_normal_cdf, normalized_gaussian};

const SIGMA: f64 = 3.0;
const BINS: usize = 256;
const RANGE: f64 = 150.0;
const CENTER: f64 = 89.5;
const THREADS: Option<usize> = Some(0);

fn bench_inverse_normal_cdf(c: &mut Criterion) {
    c.bench_function("inverse_normal_cdf", |b| {
        b.iter(|| {
            let _ = inverse_normal_cdf(0.5);
        });
    });
}

fn bench_normalized_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalized_gaussian");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = normalized_gaussian(SIGMA, BINS, RANGE, CENTER, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = normalized_gaussian(SIGMA, BINS, RANGE, CENTER, Some(1));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_inverse_normal_cdf, bench_normalized_gaussian,);
criterion_main!(benches);
