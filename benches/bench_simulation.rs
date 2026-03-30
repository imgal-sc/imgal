use divan::Bencher;

use imgal::simulation::decay;
use imgal::simulation::noise;

const SIZES: [usize; 3] = [128, 256, 512];
const SAMPLES: usize = 256;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;

fn main() {
    divan::main();
}

#[divan::bench(args = SIZES)]
fn bencher_simulate_poisson_noise_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            decay::gaussian_exponential_decay_3d(
                SAMPLES,
                PERIOD,
                &TAUS,
                &FRACTIONS,
                TOTAL_COUNTS,
                IRF_CENTER,
                IRF_WIDTH,
                (size, size),
            )
            .unwrap()
        })
        .bench_values(|v| {
            let _ = noise::poisson_noise(&v, 0.8, None, false);
        })
}

#[divan::bench(args = SIZES)]
fn bencher_simulate_poisson_noise_mut_sequential(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            decay::gaussian_exponential_decay_3d(
                SAMPLES,
                PERIOD,
                &TAUS,
                &FRACTIONS,
                TOTAL_COUNTS,
                IRF_CENTER,
                IRF_WIDTH,
                (size, size),
            )
            .unwrap()
        })
        .bench_values(|mut v| {
            noise::poisson_noise_mut(v.view_mut().into_dyn(), 0.8, None, false);
        })
}
