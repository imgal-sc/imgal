use divan::Bencher;
use ndarray::Array2;
use rand::Rng;

use imgal::spatial::KDTree;

fn main() {
    // run registered benchmarks
    divan::main();
}

/// Generate a f64 random n-dimensional point cloud
fn point_cloud(n_points: usize, n_dims: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let mut cloud: Array2<f64> = Array2::zeros((n_points, n_dims));
    cloud
        .iter_mut()
        .for_each(|d| *d = rng.random_range(0.0..1000.0));

    cloud
}

#[divan::bench(args= [100, 1000, 100_000])]
fn build_kdtree_3d(b: Bencher, n_points: usize) {
    let cloud = point_cloud(n_points, 3);

    b.bench(|| {
        let tree = KDTree::build(&cloud);
        divan::black_box(tree);
    });
}

#[divan::bench(args= [100, 1000, 100_000])]
fn search_kdtree_3d(b: Bencher, n_points: usize) {
    let cloud = point_cloud(n_points, 3);
    let tree = KDTree::build(&cloud);
    let query = [3.0, 4.0, 5.0];

    b.bench(|| {
        let indices = tree.search_for_indices(&query, 10.0).unwrap();
        divan::black_box(indices);
    });
}
