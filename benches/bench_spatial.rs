use criterion::{Criterion, criterion_group, criterion_main};
use imgal::spatial::geometry::{inside_polyhedron, inside_tetrahedron, orient_pred_3d};
use ndarray::{Array2, arr1};

use imgal::constants::RNG_SEED;
use imgal::simulation::rng::Pcg;
use imgal::spatial::KDTree;
use imgal::spatial::convex_hull::quickhull_3d;
use imgal::spatial::geometry::hull_centroid;

const THREADS: Option<usize> = Some(0);

fn bench_kdtree(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree");
    let mut prng = Pcg::new(RNG_SEED);
    let mut cloud: Array2<u32> = Array2::zeros((1_000_000, 3));
    cloud
        .iter_mut()
        .for_each(|v| *v = prng.next_u32_range(0..1000).unwrap());
    group.bench_function("build", |b| {
        b.iter(|| {
            let _ = KDTree::build(&cloud);
        });
    });
    let tree = KDTree::build(&cloud);
    let query = [32, 83, 10];
    group.bench_function("search_for_indices", |b| {
        b.iter(|| {
            let _ = tree.search_for_indices(&query, 10.0).unwrap();
        });
    });
}

fn bench_inside_polyhedron(c: &mut Criterion) {
    let mut group = c.benchmark_group("inside_polyhedron");
    let mut cloud = Array2::<f32>::zeros((10_000, 3));
    let mut prng = Pcg::new(RNG_SEED);
    let query = arr1(&[prng.next_f32(), prng.next_f32(), prng.next_f32()]);
    cloud.iter_mut().for_each(|v| *v = prng.next_f32());
    let (verts, faces) = quickhull_3d(&cloud, Some(1)).unwrap();
    let center = hull_centroid(&cloud, Some(1)).unwrap().mapv(|v| v as f32);
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = inside_polyhedron(&verts, &faces, &center, &query, Some(1));
        });
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = inside_polyhedron(&verts, &faces, &center, &query, THREADS);
        });
    });
}

fn bench_inside_tetrahedron(c: &mut Criterion) {
    let pnt_a = arr1(&[3.2, 0.4, 8.5]);
    let pnt_b = arr1(&[6.7, 1.1, 9.8]);
    let pnt_c = arr1(&[0.0, 4.9, 5.1]);
    let pnt_d = arr1(&[0.0, 1.2, 8.0]);
    let query = arr1(&[2.5, 1.8, 7.9]);
    c.bench_function("inside_tetrahedron", |b| {
        b.iter(|| {
            let _ = inside_tetrahedron(&pnt_a, &pnt_b, &pnt_c, &pnt_d, &query).unwrap();
        })
    });
}

fn bench_orient_pred_3d(c: &mut Criterion) {
    let pnt_a = arr1(&[3.2, 0.4, 8.5]);
    let pnt_b = arr1(&[6.7, 1.1, 9.8]);
    let pnt_c = arr1(&[0.0, 4.9, 5.1]);
    let pnt_d = arr1(&[0.0, 1.2, 8.0]);
    c.bench_function("orient_pred_3d", |b| {
        b.iter(|| {
            let _ = orient_pred_3d(&pnt_a, &pnt_b, &pnt_c, &pnt_d).unwrap();
        })
    });
}

fn bench_quickhull_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("quickhull_3d");
    let mut cloud = Array2::<f32>::zeros((50_000, 3));
    let mut prng = Pcg::new(RNG_SEED);
    cloud.iter_mut().for_each(|v| *v = prng.next_f32());
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = quickhull_3d(&cloud, Some(1));
        });
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = quickhull_3d(&cloud, THREADS);
        });
    });
}

criterion_group!(
    benches,
    bench_kdtree,
    bench_inside_polyhedron,
    bench_inside_tetrahedron,
    bench_orient_pred_3d,
    bench_quickhull_3d
);
criterion_main!(benches);
