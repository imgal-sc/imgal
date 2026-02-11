use divan;
use ndarray::{Array2, ArrayD, Ix2, Ix3};

use imgal::colocalization::{saca_2d, saca_3d, saca_significance_mask};
use imgal::simulation::blob::logistic_metaballs;
use imgal::threshold::global::otsu_value;

fn main() {
    divan::main();
}

fn sim_coloc_data(size: usize, n_dims: usize) -> (ArrayD<f64>, ArrayD<f64>) {
    // logistic metaball simulation parameters, blob "A" is centered on the
    // image and blob "B" is offset by 5 pixels
    let a_pos = (size / 2) as u16;
    let b_pos = a_pos + 5;
    let center_a = Array2::from_shape_vec((1, n_dims), vec![a_pos; n_dims]).unwrap();
    let center_b = Array2::from_shape_vec((1, n_dims), vec![b_pos; n_dims]).unwrap();
    let shape: Vec<usize>;
    if n_dims == 2 {
        shape = vec![size; n_dims];
    } else {
        shape = vec![size, size, 15];
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
    )
    .unwrap();
    let blob_b = logistic_metaballs(
        &center_b,
        &radii,
        &intensities,
        &falloffs,
        background,
        &shape,
    )
    .unwrap();

    (blob_a, blob_b)
}

// Benchmark the colocalization namespace.
#[divan::bench(args = [64, 128, 256], sample_count = 5, sample_size = 5)]
fn bench_saca_2d_parallel(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let (a, b) = sim_coloc_data(size, 2);
            let a = a.into_dimensionality::<Ix2>().unwrap();
            let b = b.into_dimensionality::<Ix2>().unwrap();
            let ta = otsu_value(&a, None, false).unwrap();
            let tb = otsu_value(&b, None, false).unwrap();
            (a, b, ta, tb)
        })
        .bench_values(|(a, b, ta, tb)| {
        let _res = saca_2d(&a, &b, ta, tb, true).unwrap();
    });
}

#[divan::bench(args = [64, 128, 256], sample_count = 5, sample_size = 5)]
fn bench_saca_3d_parallel(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let (a, b) = sim_coloc_data(size, 3);
            let a = a.into_dimensionality::<Ix3>().unwrap();
            let b = b.into_dimensionality::<Ix3>().unwrap();
            let ta = otsu_value(&a, None, false).unwrap();
            let tb = otsu_value(&b, None, false).unwrap();
            (a, b, ta, tb)
        })
        .bench_values(|(a, b, ta, tb)| {
        let _res = saca_3d(&a, &b, ta, tb, true).unwrap();
    });
}

#[divan::bench(args = [64, 128, 256], sample_count = 5, sample_size = 5)]
fn bench_saca_significance_mask_parallel(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let (a, b) = sim_coloc_data(size, 3);
            let a = a.into_dimensionality::<Ix3>().unwrap();
            let b = b.into_dimensionality::<Ix3>().unwrap();
            let ta = otsu_value(&a, None, false).unwrap();
            let tb = otsu_value(&b, None, false).unwrap();
            saca_3d(&a, &b, ta, tb, true).unwrap()
        })
        .bench_values(|z| {
        let _res = saca_significance_mask(&z, None, true);
    });
}

#[divan::bench(args = [64, 128, 256], sample_count = 5, sample_size = 5)]
fn bench_saca_2d_sequential(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let (a, b) = sim_coloc_data(size, 2);
            let a = a.into_dimensionality::<Ix2>().unwrap();
            let b = b.into_dimensionality::<Ix2>().unwrap();
            let ta = otsu_value(&a, None, false).unwrap();
            let tb = otsu_value(&b, None, false).unwrap();
            (a, b, ta, tb)
        })
        .bench_values(|(a, b, ta, tb)| {
        let _res = saca_2d(&a, &b, ta, tb, false).unwrap();
    });
}

#[divan::bench(args = [64, 128, 256], sample_count = 5, sample_size = 5)]
fn bench_saca_3d_sequential(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let (a, b) = sim_coloc_data(size, 3);
            let a = a.into_dimensionality::<Ix3>().unwrap();
            let b = b.into_dimensionality::<Ix3>().unwrap();
            let ta = otsu_value(&a, None, false).unwrap();
            let tb = otsu_value(&b, None, false).unwrap();
            (a, b, ta, tb)
        })
        .bench_values(|(a, b, ta, tb)| {
        let _res = saca_3d(&a, &b, ta, tb, false).unwrap();
    });
}

#[divan::bench(args = [64, 128, 256], sample_count = 5, sample_size = 5)]
fn bench_saca_significance_mask_sequential(bencher: divan::Bencher, size: usize) {
    bencher
        .with_inputs(|| {
            let (a, b) = sim_coloc_data(size, 3);
            let a = a.into_dimensionality::<Ix3>().unwrap();
            let b = b.into_dimensionality::<Ix3>().unwrap();
            let ta = otsu_value(&a, None, false).unwrap();
            let tb = otsu_value(&b, None, false).unwrap();
            saca_3d(&a, &b, ta, tb, true).unwrap()
        })
        .bench_values(|z| {
        let _res = saca_significance_mask(&z, None, false);
    });
}
