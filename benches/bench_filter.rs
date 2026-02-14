use divan::Bencher;

use imgal::filter::{fft_convolve_1d, fft_deconvolve_1d};
use imgal::simulation::{decay, instrument};

const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;

fn main() {
    divan::main();
}

#[divan::bench(args = [5000, 500_000, 5_000_000], sample_count = 5, sample_size = 5)]
fn bench_fft_convolve_1d_parallel(bencher: Bencher, samples: usize) {
    bencher
        .with_inputs(|| {
            let a =
                decay::ideal_exponential_decay_1d(samples, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)
                    .unwrap();
            let b = instrument::gaussian_irf_1d(samples, PERIOD, IRF_CENTER, IRF_WIDTH);
            (a, b)
        })
        .bench_values(|(a, b)| {
            let _ = fft_convolve_1d(&a, &b, true);
        });
}

#[divan::bench(args = [5000, 500_000, 5_000_000], sample_count = 5, sample_size = 5)]
fn bench_fft_deconvolve_1d_parallel(bencher: Bencher, samples: usize) {
    bencher
        .with_inputs(|| {
            let a = decay::gaussian_exponential_decay_1d(
                samples,
                PERIOD,
                &TAUS,
                &FRACTIONS,
                TOTAL_COUNTS,
                IRF_CENTER,
                IRF_WIDTH,
            )
            .unwrap();
            let b =
                decay::ideal_exponential_decay_1d(samples, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)
                    .unwrap();
            (a, b)
        })
        .bench_values(|(a, b)| {
            let _ = fft_deconvolve_1d(&a, &b, None, true);
        });
}

#[divan::bench(args = [5000, 500_000, 5_000_000], sample_count = 5, sample_size = 5)]
fn bench_fft_convolve_1d_sequential(bencher: Bencher, samples: usize) {
    bencher
        .with_inputs(|| {
            let a =
                decay::ideal_exponential_decay_1d(samples, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)
                    .unwrap();
            let b = instrument::gaussian_irf_1d(samples, PERIOD, IRF_CENTER, IRF_WIDTH);
            (a, b)
        })
        .bench_values(|(a, b)| {
            let _ = fft_convolve_1d(&a, &b, false);
        });
}

#[divan::bench(args = [5000, 500_000, 5_000_000], sample_count = 5, sample_size = 5)]
fn bench_fft_deconvolve_1d_sequential(bencher: Bencher, samples: usize) {
    bencher
        .with_inputs(|| {
            let a = decay::gaussian_exponential_decay_1d(
                samples,
                PERIOD,
                &TAUS,
                &FRACTIONS,
                TOTAL_COUNTS,
                IRF_CENTER,
                IRF_WIDTH,
            )
            .unwrap();
            let b =
                decay::ideal_exponential_decay_1d(samples, PERIOD, &TAUS, &FRACTIONS, TOTAL_COUNTS)
                    .unwrap();
            (a, b)
        })
        .bench_values(|(a, b)| {
            let _ = fft_deconvolve_1d(&a, &b, None, false);
        });
}
