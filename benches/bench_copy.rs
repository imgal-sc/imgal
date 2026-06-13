use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::arr2;

use imgal::copy::{copy_into_flat, duplicate};
use imgal::simulation::blob::gaussian_metaballs;

const SIZES: [usize; 3] = [256, 512, 1024];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const THREADS: Option<usize> = Some(0);

fn bench_copy_into_flat_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_into_flat (Parallel)");
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
                let _ = copy_into_flat(&data, THREADS);
            });
        });
    });
    group.finish();
}

fn bench_copy_into_flat_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_into_flat (Sequential)");
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
                let _ = copy_into_flat(&data, None);
            });
        });
    });
    group.finish();
}

fn bench_duplicate_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("duplicate (Parallel)");
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
                let _ = duplicate(&data, THREADS);
            });
        });
    });
    group.finish();
}

fn bench_duplicate_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("duplicate (Sequential)");
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
                let _ = duplicate(&data, None);
            });
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_copy_into_flat_par,
    bench_copy_into_flat_seq,
    bench_duplicate_par,
    bench_duplicate_seq
);
criterion_main!(benches);
