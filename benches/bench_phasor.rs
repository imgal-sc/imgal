use divan::Bencher;

use imgal::phasor::time_domain;
use imgal::simulation::decay;

// simulated bioexponential decay parameters
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

#[divan::bench(args = [256, 512, 1024])]
fn bench_phasor_gs_image(bencher: Bencher, size: usize) {
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
            let _ = time_domain::gs_image(&v, PERIOD, None, None, None).unwrap();
        });
}
