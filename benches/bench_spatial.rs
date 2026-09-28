use criterion::{Criterion, criterion_group, criterion_main};
use imgal::spatial::geometry::{inside_polyhedron, inside_tetrahedron, orient_pred_3d};
use ndarray::{Array2, arr1, arr2, array};

use imgal::constants::RNG_SEED;
use imgal::simulation::rng::Pcg;
use imgal::spatial::KDTree;
use imgal::spatial::convex_hull::{graham_scan, quickhull_3d};
use imgal::spatial::geometry::hull_centroid;
use imgal::spatial::halfplane::{
    edge_to_halfplane, halfplane_intersection, hull_to_halfplane, inside_halfplane_interior,
};
use imgal::spatial::halfspace::{
    face_to_halfspace, halfspace_intersection, hull_to_halfspace, inside_halfspace_interior,
};

const HALFPLANES: [[f64; 3]; 4] = [
    [-3.5, 1.6, -9.81],
    [1.4, 4.5, -19.72],
    [5.1, -1.4, -14.9],
    [-3.0, -4.7, -1.73],
];
const THREADS: Option<usize> = Some(0);

fn bench_edge_to_halfplane(c: &mut Criterion) {
    let a_verts = [-2.3, 1.1];
    let b_verts = [-0.7, 4.6];
    c.bench_function("edge_to_halfplane", |b| {
        b.iter(|| {
            let _ = edge_to_halfplane(&a_verts, &b_verts);
        })
    });
}

fn bench_face_to_halfspace(c: &mut Criterion) {
    let a_verts = array![1.0, 2.0, 3.0];
    let b_verts = array![4.0, 0.0, 1.0];
    let c_verts = array![0.0, 3.0, 5.0];
    c.bench_function("face_to_halfspace", |b| {
        b.iter(|| {
            let _ = face_to_halfspace(&a_verts, &b_verts, &c_verts);
        })
    });
}

fn bench_halfplane_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("halfplane_intersection");
    let query = [0.5, 1.0];
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = halfplane_intersection(&HALFPLANES, &query, Some(1));
        });
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = halfplane_intersection(&HALFPLANES, &query, THREADS);
        });
    });
    group.finish();
}

fn bench_halfspace_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("halfspace_intersection");
    let oct_hs = arr2(&[
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0],
    ]);
    let oct_interior = array![0.0, 0.0, 0.0];
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = halfspace_intersection(&oct_hs, &oct_interior, Some(1));
        });
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = halfspace_intersection(&oct_hs, &oct_interior, THREADS);
        })
    });
    group.finish();
}

fn bench_hull_to_halfplane(c: &mut Criterion) {
    let mut group = c.benchmark_group("hull_to_halfplane");
    let mut cloud = Array2::<f32>::zeros((100_000, 2));
    let mut prng = Pcg::new(RNG_SEED);
    cloud.iter_mut().for_each(|v| *v = prng.next_f32());
    let verts = graham_scan(&cloud, Some(1)).unwrap();
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = hull_to_halfplane(&verts, Some(1));
        });
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = hull_to_halfplane(&verts, THREADS);
        });
    });
    group.finish();
}

fn bench_hull_to_halfspace(c: &mut Criterion) {
    let mut group = c.benchmark_group("hull_to_halfspace");
    let mut cloud = Array2::<f32>::zeros((100_000, 3));
    let mut prng = Pcg::new(RNG_SEED);
    cloud.iter_mut().for_each(|v| *v = prng.next_f32());
    let (verts, faces) = quickhull_3d(&cloud, Some(1)).unwrap();
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = hull_to_halfspace(&verts, &faces, Some(1));
        });
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = hull_to_halfspace(&verts, &faces, THREADS);
        });
    });
    group.finish();
}

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
    group.finish();
}

fn bench_inside_halfplane_interior(c: &mut Criterion) {
    let mut group = c.benchmark_group("inside_halfplane_interior");
    let query = [0.5, 1.0];
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = inside_halfplane_interior(&HALFPLANES, &query, true, Some(1));
        });
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = inside_halfplane_interior(&HALFPLANES, &query, true, THREADS);
        });
    });
    group.finish();
}

fn bench_inside_halfspace_interior(c: &mut Criterion) {
    let mut group = c.benchmark_group("inside_halfspace_interior");
    let cube_hs = arr2(&[
        [1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, -1.0, -1.0],
    ]);
    let inside = array![0.0, 0.0, 0.0];
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            let _ = inside_halfspace_interior(&cube_hs, &inside, true, Some(1));
        })
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            let _ = inside_halfspace_interior(&cube_hs, &inside, true, THREADS);
        })
    });
    group.finish();
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
    group.finish();
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
    let mut cloud = Array2::<f32>::zeros((100_000, 3));
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
    group.finish();
}

criterion_group!(
    benches,
    bench_edge_to_halfplane,
    bench_face_to_halfspace,
    bench_halfplane_intersection,
    bench_halfspace_intersection,
    bench_hull_to_halfplane,
    bench_hull_to_halfspace,
    bench_kdtree,
    bench_inside_halfplane_interior,
    bench_inside_halfspace_interior,
    bench_inside_polyhedron,
    bench_inside_tetrahedron,
    bench_orient_pred_3d,
    bench_quickhull_3d
);
criterion_main!(benches);
