use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::arr2;

use imgal::image::histogram;
use imgal::simulation::blob::gaussian_metaballs;

const SIZES: [usize; 3] = [256, 512, 1024];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const THREADS: Option<usize> = Some(0);

fn bench_histogram_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram (Parallel)");
    SIZES.iter().for_each(|s| {
        group.bench_with_input(BenchmarkId::from_parameter(s), s, |b, &s| {
            let center: [[f64; 2]; 1] = [[s as f64 / 2.0, s as f64 / 2.0]];
            let shape: [usize; 2] = [s, s];
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
            b.iter(|| {
                let _ = histogram(&data, None, THREADS).unwrap();
            });
        });
    });
}

fn bench_histogram_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram (Sequential)");
    SIZES.iter().for_each(|s| {
        group.bench_with_input(BenchmarkId::from_parameter(s), s, |b, &s| {
            let center: [[f64; 2]; 1] = [[s as f64 / 2.0, s as f64 / 2.0]];
            let shape: [usize; 2] = [s, s];
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
            b.iter(|| {
                let _ = histogram(&data, None, None).unwrap();
            });
        });
    });
}

criterion_group!(benches, bench_histogram_par, bench_histogram_seq);
criterion_main!(benches);
