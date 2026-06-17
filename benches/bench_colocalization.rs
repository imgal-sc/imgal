use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::{Array2, ArrayD, Ix2, Ix3};

use imgal::colocalization::{saca_2d, saca_3d, saca_significance_mask};
use imgal::prelude::*;
use imgal::simulation::blob::logistic_metaballs;
use imgal::threshold::global::otsu_value;

const SIZE: usize = 32;
const THREADS: Option<usize> = Some(0);

fn sim_coloc_data(size: usize, n_dims: usize) -> Result<(ArrayD<f64>, ArrayD<f64>), ImgalError> {
    let a_pos = (size / 2) as u16;
    let b_pos = a_pos + 5;
    let center_a = Array2::from_shape_vec((1, n_dims), vec![a_pos; n_dims]).unwrap();
    let center_b = Array2::from_shape_vec((1, n_dims), vec![b_pos; n_dims]).unwrap();
    let shape: Vec<usize>;
    if n_dims == 2 {
        shape = vec![size; n_dims];
    } else {
        shape = vec![3, size, size];
    }
    let radii = vec![(size / 4) as u16];
    let intensities = vec![20_u16];
    let falloffs = vec![3_u16];
    let background = 0;
    let blob_a = logistic_metaballs(
        &center_a,
        &radii,
        &intensities,
        &falloffs,
        background,
        &shape,
        None,
    )?;
    let blob_b = logistic_metaballs(
        &center_b,
        &radii,
        &intensities,
        &falloffs,
        background,
        &shape,
        None,
    )?;
    Ok((blob_a, blob_b))
}

fn bench_saca_2d(c: &mut Criterion) {
    let (ch_a, ch_b) = sim_coloc_data(SIZE, 2).unwrap();
    let ch_a = ch_a.into_dimensionality::<Ix2>().unwrap();
    let ch_b = ch_b.into_dimensionality::<Ix2>().unwrap();
    let ta = otsu_value(&ch_a, None, None).unwrap();
    let tb = otsu_value(&ch_b, None, None).unwrap();
    let mut group = c.benchmark_group("saca_2d");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = saca_2d(&ch_a, &ch_b, ta, tb, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = saca_2d(&ch_a, &ch_b, ta, tb, Some(1)).unwrap();
        });
    });
    group.finish();
}

fn bench_saca_3d(c: &mut Criterion) {
    let (ch_a, ch_b) = sim_coloc_data(SIZE, 3).unwrap();
    let ch_a = ch_a.into_dimensionality::<Ix3>().unwrap();
    let ch_b = ch_b.into_dimensionality::<Ix3>().unwrap();
    let ta = otsu_value(&ch_a, None, None).unwrap();
    let tb = otsu_value(&ch_b, None, None).unwrap();
    let mut group = c.benchmark_group("saca_3d");
    group.sample_size(10);
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = saca_3d(&ch_a, &ch_b, ta, tb, THREADS).unwrap();
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = saca_3d(&ch_a, &ch_b, ta, tb, Some(1)).unwrap();
        });
    });
    group.finish();
}

fn bench_saca_significance_mask(c: &mut Criterion) {
    let (ch_a, ch_b) = sim_coloc_data(SIZE, 2).unwrap();
    let ch_a = ch_a.into_dimensionality::<Ix2>().unwrap();
    let ch_b = ch_b.into_dimensionality::<Ix2>().unwrap();
    let ta = otsu_value(&ch_a, None, None).unwrap();
    let tb = otsu_value(&ch_b, None, None).unwrap();
    let z = saca_2d(&ch_a, &ch_b, ta, tb, Some(0)).unwrap();
    let mut group = c.benchmark_group("saca_significance_mask");
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = saca_significance_mask(&z, None, THREADS);
        });
    });
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = saca_significance_mask(&z, None, Some(1));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_saca_2d,
    bench_saca_3d,
    bench_saca_significance_mask,
);
criterion_main!(benches);
