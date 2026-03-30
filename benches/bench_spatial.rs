use divan::Bencher;
use ndarray::Array2;

use imgal::constants::RNG_SEED;
use imgal::simulation::rng::Pcg;
use imgal::spatial::KDTree;

fn main() {
    // run registered benchmarks
    divan::main();
}

/// Generate a u32 random n-dimensional point cloud
fn point_cloud(n_points: usize, n_dims: usize) -> Array2<u32> {
    let mut prng = Pcg::new(RNG_SEED);
    let mut cloud: Array2<u32> = Array2::zeros((n_points, n_dims));
    cloud
        .iter_mut()
        .for_each(|d| *d = prng.next_u32_range(0..1000).unwrap());

    cloud
}

#[divan::bench(args= [1000, 100_000, 1_000_000])]
fn build_kdtree_3d(bencher: Bencher, n_points: usize) {
    bencher
        .with_inputs(|| point_cloud(n_points, 3))
        .bench_values(|c| {
            divan::black_box(KDTree::build(&c));
        });
}

#[divan::bench(args= [1000, 100_000, 1_000_000])]
fn search_kdtree_3d(bencher: Bencher, n_points: usize) {
    let cloud = point_cloud(n_points, 3);
    bencher
        .with_inputs(|| {
            let tree = KDTree::build(&cloud);
            let query = [32, 83, 10];
            (tree, query)
        })
        .bench_values(|(t, q)| {
            let _indices = t.search_for_indices(&q, 10.0).unwrap();
        });
}
