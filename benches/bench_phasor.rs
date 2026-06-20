use criterion::{Criterion, criterion_group, criterion_main};

use imgal::phasor::time_domain::gs_image;
use imgal::simulation::decay::gaussian_exponential_decay_3d;

const SAMPLES: usize = 256;
const PERIOD: f64 = 12.5;
const TAUS: [f64; 2] = [1.0, 3.0];
const FRACTIONS: [f64; 2] = [0.7, 0.3];
const TOTAL_COUNTS: f64 = 5000.0;
const IRF_CENTER: f64 = 3.0;
const IRF_WIDTH: f64 = 0.5;
const SHAPE: (usize, usize) = (1024, 1024);
const THREADS: Option<usize> = Some(0);

fn bench_gs_image(c: &mut Criterion) {
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
    let mut group = c.benchmark_group("gs_image");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = gs_image(&data, PERIOD, None, None, None, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = gs_image(&data, PERIOD, None, None, None, Some(1)).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_gs_image);
criterion_main!(benches);
