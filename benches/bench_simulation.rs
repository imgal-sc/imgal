use criterion::{Criterion, criterion_group, criterion_main};

use imgal::simulation::decay::gaussian_exponential_decay_3d;
use imgal::simulation::noise::{poisson_noise, poisson_noise_mut};

const SAMPLES: usize = 256;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;
const SHAPE: (usize, usize) = (1024, 1024);
const THREADS: Option<usize> = Some(0);

fn bench_poisson_noise(c: &mut Criterion) {
    let data = gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        SHAPE,
        None,
    )
    .unwrap();
    let mut group = c.benchmark_group("poisson_noise");
    group.sample_size(10);
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = poisson_noise(&data, 0.8, None, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = poisson_noise(&data, 0.8, None, Some(1));
        });
    });
    group.finish();
}

fn bench_poisson_noise_mut(c: &mut Criterion) {
    let mut data = gaussian_exponential_decay_3d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        SHAPE,
        None,
    )
    .unwrap();
    let mut group = c.benchmark_group("poisson_noise_mut");
    group.sample_size(10);
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            poisson_noise_mut(data.view_mut().into_dyn(), 0.8, None, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            poisson_noise_mut(data.view_mut().into_dyn(), 0.8, None, Some(1));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_poisson_noise, bench_poisson_noise_mut);
criterion_main!(benches);
