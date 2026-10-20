use criterion::{Criterion, criterion_group, criterion_main};

use imgal::filter::{fft_convolve_1d, fft_deconvolve_1d};
use imgal::simulation::decay::{gaussian_exponential_decay_1d, ideal_exponential_decay_1d};
use imgal::simulation::instrument::gaussian_irf_1d;

const SAMPLES: usize = 5_000_000;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;
const THREADS: Option<usize> = Some(0);

fn bench_fft_convolve_1d(c: &mut Criterion) {
    let arr_a =
        ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, None).unwrap();
    let arr_b = gaussian_irf_1d(SAMPLES, PERIOD, IRF_CENTER, IRF_WIDTH, None);
    let mut group = c.benchmark_group("fft_convolve_1d");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = fft_convolve_1d(&arr_a, &arr_b, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = fft_convolve_1d(&arr_a, &arr_b, Some(1));
        });
    });
    group.finish();
}

fn bench_fft_deconvolve_1d(c: &mut Criterion) {
    let arr_a = gaussian_exponential_decay_1d(
        SAMPLES,
        PERIOD,
        &TAUS,
        &FRACTIONS,
        TOTAL_COUNTS,
        IRF_CENTER,
        IRF_WIDTH,
        None,
    )
    .unwrap();
    let arr_b =
        ideal_exponential_decay_1d(SAMPLES, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS, None).unwrap();
    let mut group = c.benchmark_group("fft_convolve_1d");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = fft_deconvolve_1d(&arr_a, &arr_b, None, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = fft_deconvolve_1d(&arr_a, &arr_b, None, Some(1));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_fft_convolve_1d, bench_fft_deconvolve_1d);
criterion_main!(benches);
