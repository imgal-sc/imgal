use divan::Bencher;
use ndarray::arr2;

use imgal::simulation::blob::gaussian_metaballs;
use imgal::threshold::{otsu_mask, otsu_value};

fn main() {
    divan::main();
}

// Benchmark the threshold namespace.
#[divan::bench(args = [256, 512, 1024])]
fn bench_threshold_otsu_mask_parallel(bencher: Bencher, size: usize) {
    let cen_pos = (size / 2) as i32;
    let center = [cen_pos, cen_pos];
    let center = arr2(&[center]);
    let radius = vec![(size / 32) as i32];
    let intensity = vec![20_i32];
    let falloff = vec![6_i32];
    let background = 0;
    let shape = vec![size, size, 30];
    let blob_sim =
        gaussian_metaballs(&center, &radius, &intensity, &falloff, background, &shape).unwrap();
    bencher.bench(|| {
        let _ = otsu_mask(&blob_sim, None, true);
    });
}

#[divan::bench(args = [256, 512, 1024])]
fn bench_threshold_otsu_value_parallel(bencher: Bencher, size: usize) {
    let cen_pos = (size / 2) as i32;
    let center = [cen_pos, cen_pos];
    let center = arr2(&[center]);
    let radius = vec![(size / 32) as i32];
    let intensity = vec![20_i32];
    let falloff = vec![6_i32];
    let background = 0;
    let shape = vec![size, size, 30];
    let blob_sim =
        gaussian_metaballs(&center, &radius, &intensity, &falloff, background, &shape).unwrap();
    bencher.bench(|| {
        let _ = otsu_mask(&blob_sim, None, true);
    });
}

#[divan::bench(args = [256, 512, 1024])]
fn bench_threshold_otsu_mask_sequential(bencher: Bencher, size: usize) {
    let cen_pos = (size / 2) as i32;
    let center = [cen_pos, cen_pos];
    let center = arr2(&[center]);
    let radius = vec![(size / 32) as i32];
    let intensity = vec![20_i32];
    let falloff = vec![6_i32];
    let background = 0;
    let shape = vec![size, size, 30];
    let blob_sim =
        gaussian_metaballs(&center, &radius, &intensity, &falloff, background, &shape).unwrap();
    bencher.bench(|| {
        let _ = otsu_mask(&blob_sim, None, false);
    });
}

#[divan::bench(args = [256, 512, 1024])]
fn bench_threshold_otsu_value_sequential(bencher: Bencher, size: usize) {
    let cen_pos = (size / 2) as i32;
    let center = [cen_pos, cen_pos];
    let center = arr2(&[center]);
    let radius = vec![(size / 32) as i32];
    let intensity = vec![20_i32];
    let falloff = vec![6_i32];
    let background = 0;
    let shape = vec![size, size, 30];
    let blob_sim =
        gaussian_metaballs(&center, &radius, &intensity, &falloff, background, &shape).unwrap();
    bencher.bench(|| {
        let _ = otsu_value(&blob_sim, None, false);
    });
}
