use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array2;

use imgal::constants::RNG_SEED;
use imgal::simulation::rng::Pcg;
use imgal::spatial::KDTree;
use imgal::spatial::convex_hull::quickhull_3d;

const N_POINTS: usize = 1_000_000;

fn point_cloud(n_points: usize, n_dims: usize) -> Array2<u32> {
    let mut prng = Pcg::new(RNG_SEED);
    let mut cloud: Array2<u32> = Array2::zeros((n_points, n_dims));
    cloud
        .iter_mut()
        .for_each(|d| *d = prng.next_u32_range(0..1000).unwrap());

    cloud
}

fn bench_kdtree(c: &mut Criterion) {
    let data = point_cloud(N_POINTS, 3);
    let mut group = c.benchmark_group("kdtree");
    group.bench_function("build", |b| {
        b.iter(|| {
            let _ = KDTree::build(&data);
        });
    });
    let tree = KDTree::build(&data);
    let query = [32, 83, 10];
    group.bench_function("search_for_indices", |b| {
        b.iter(|| {
            let _ = tree.search_for_indices(&query, 10.0).unwrap();
        });
    });
}

fn bench_quickhull_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("quickhull_3d");
    let mut cloud = Array2::<f32>::zeros((10_000, 3));
    let mut prng = Pcg::new(RNG_SEED);
    cloud.iter_mut().for_each(|v| *v = prng.next_f32());
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = quickhull_3d(&cloud, Some(1));
        });
    });
}

criterion_group!(benches, bench_kdtree, bench_quickhull_3d);
criterion_main!(benches);
