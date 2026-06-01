use imgal::spatial::geometry::tetrahedron_volume;
use ndarray::{Array1, arr2, array, s};

use imgal::ImgalError;
use imgal::spatial::KDTree;
use imgal::spatial::convex_hull::{chan_2d, graham_scan, jarvis_march, quickhull_3d};
use imgal::spatial::halfspace::face_to_halfspace;

const TOLERANCE: f64 = 1e-10;
const POINTS_2D: [[f64; 2]; 12] = [
    [-3.9, 5.8],
    [-4.7, 8.1],
    [-1.2, 9.4],
    [3.6, 7.2],
    [-4.7, 2.3],
    [5.2, 3.1],
    [3.9, -0.8],
    [0.4, -2.5],
    [-1.3, 1.7],
    [-0.2, 4.6],
    [7.9, 9.9],
    [-11.3, 3.4],
];
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Tests that `chan_2d` returns the expected convex hull with the start point
/// at index `0` and hull size.
#[test]
fn convex_hull_chan_2d_expected_results() -> Result<(), ImgalError> {
    let points = arr2(&POINTS_2D);
    let hull_par = chan_2d(&points, THREADS)?;
    let hull_seq = chan_2d(&points, None)?;
    assert_eq!(hull_par.slice(s![0, ..]), array![0.4, -2.5]);
    assert_eq!(hull_seq.slice(s![0, ..]), array![0.4, -2.5]);
    assert_eq!(hull_par.dim().0, 6);
    assert_eq!(hull_seq.dim().0, 6);
    Ok(())
}

/// Tests that `graham_scan` returns the expected convex hull with the pivot
/// point at index `0` and hull size.
#[test]
fn convex_hull_graham_scan_expected_results() -> Result<(), ImgalError> {
    let points = arr2(&POINTS_2D);
    let hull_par = graham_scan(&points, THREADS)?;
    let hull_seq = graham_scan(&points, None)?;
    assert_eq!(hull_par.slice(s![0, ..]), array![-11.3, 3.4]);
    assert_eq!(hull_seq.slice(s![0, ..]), array![-11.3, 3.4]);
    assert_eq!(hull_par.dim().0, 6);
    assert_eq!(hull_seq.dim().0, 6);
    Ok(())
}

/// Tests that `jarvis_march` returns the expected convex hull with the start
/// point at index `0` and hull size.
#[test]
fn convex_hull_jarvis_march_expected_results() -> Result<(), ImgalError> {
    let points = arr2(&POINTS_2D);
    let hull_par = jarvis_march(&points, THREADS)?;
    let hull_seq = jarvis_march(&points, None)?;
    assert_eq!(hull_par.slice(s![0, ..]), array![0.4, -2.5]);
    assert_eq!(hull_seq.slice(s![0, ..]), array![0.4, -2.5]);
    assert_eq!(hull_par.dim().0, 6);
    assert_eq!(hull_seq.dim().0, 6);
    Ok(())
}

/// Tests that `quickhull_3d` returns a 3D convex hull with the expected number
/// of vertices and faces.
#[test]
fn convex_hull_quickhull_3d_expected_results() -> Result<(), ImgalError> {
    let cube = arr2(&[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]);
    let cube_with_inside = arr2(&[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.2, 0.3, 0.7],
    ]);
    let octahedron = arr2(&[
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]);
    let tetrahedron = arr2(&[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let icosahedron = arr2(&[
        [0.0, 1.0, phi],
        [0.0, -1.0, phi],
        [0.0, 1.0, -phi],
        [0.0, -1.0, -phi],
        [1.0, phi, 0.0],
        [-1.0, phi, 0.0],
        [1.0, -phi, 0.0],
        [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0],
        [phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
    ]);
    let cube_hull_par = quickhull_3d(&cube, THREADS)?;
    let cube_hull_seq = quickhull_3d(&cube, None)?;
    let cube_with_inside_hull_par = quickhull_3d(&cube_with_inside, THREADS)?;
    let cube_with_inside_hull_seq = quickhull_3d(&cube_with_inside, None)?;
    let icosohedron_hull_par = quickhull_3d(&icosahedron, THREADS)?;
    let icosohedron_hull_seq = quickhull_3d(&icosahedron, None)?;
    let oct_hull_par = quickhull_3d(&octahedron, THREADS)?;
    let oct_hull_seq = quickhull_3d(&octahedron, None)?;
    let tet_hull_par = quickhull_3d(&tetrahedron, THREADS)?;
    let tet_hull_seq = quickhull_3d(&tetrahedron, None)?;
    assert_eq!(cube_hull_par.0.dim().0, 8);
    assert_eq!(cube_hull_seq.0.dim().0, 8);
    assert_eq!(cube_hull_par.1.dim().0, 12);
    assert_eq!(cube_hull_seq.1.dim().0, 12);
    assert_eq!(cube_with_inside_hull_par.0.dim().0, 8);
    assert_eq!(cube_with_inside_hull_seq.0.dim().0, 8);
    assert_eq!(cube_with_inside_hull_par.1.dim().0, 12);
    assert_eq!(cube_with_inside_hull_seq.1.dim().0, 12);
    assert_eq!(icosohedron_hull_par.0.dim().0, 12);
    assert_eq!(icosohedron_hull_seq.0.dim().0, 12);
    assert_eq!(icosohedron_hull_par.1.dim().0, 20);
    assert_eq!(icosohedron_hull_seq.1.dim().0, 20);
    assert_eq!(oct_hull_par.0.dim().0, 6);
    assert_eq!(oct_hull_seq.0.dim().0, 6);
    assert_eq!(oct_hull_par.1.dim().0, 8);
    assert_eq!(oct_hull_seq.1.dim().0, 8);
    assert_eq!(tet_hull_par.0.dim().0, 4);
    assert_eq!(tet_hull_seq.0.dim().0, 4);
    assert_eq!(tet_hull_par.1.dim().0, 4);
    assert_eq!(tet_hull_seq.1.dim().0, 4);
    Ok(())
}

/// Tests that `tetrahedron_volume` returns the expected signed tetrahedron
/// volumes.
#[test]
fn geometry_tetrahedron_volume_expected_results() -> Result<(), ImgalError> {
    let neg_vol_tet = arr2(&[
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
    ]);
    let pos_vol_tet = arr2(&[
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
    ]);
    assert!(approx_equal(
        tetrahedron_volume(
            neg_vol_tet.row(0),
            neg_vol_tet.row(1),
            neg_vol_tet.row(2),
            neg_vol_tet.row(3)
        )?,
        -2.6666666666
    ));
    assert!(approx_equal(
        tetrahedron_volume(
            pos_vol_tet.row(0),
            pos_vol_tet.row(1),
            pos_vol_tet.row(2),
            pos_vol_tet.row(3)
        )?,
        2.6666666666
    ));
    Ok(())
}

/// Tests that `face_to_halfspace` returns the expected halfspace normal vector
/// values.
#[test]
fn halfspace_face_to_halfspace_expected_results() -> Result<(), ImgalError> {
    let a = array![1.0, 2.0, 3.0];
    let b = array![4.0, 0.0, 1.0];
    let c = array![0.0, 3.0, 5.0];
    let hs = face_to_halfspace(&a, &b, &c)?;
    assert_eq!(hs, Array1::from_vec(vec![2.0, 4.0, -1.0, -7.0]));
    Ok(())
}

/// Tests that `KDTree` can be constructed and returns the expected values when
/// searching for coordinates and indices.
#[test]
fn spatial_kdtree_expected_results() -> Result<(), ImgalError> {
    let cloud = array![
        [-2.7, 3.9, 5.0],
        [0.1, 0.0, 4.0],
        [1.4, 0.2, 2.1],
        [-3.2, -1.8, -2.3],
        [-4.9, -3.7, -1.1],
    ];
    let tree = KDTree::build(&cloud);
    let query = [0.0, 0.0, 0.0];
    let result_inds = tree.search_for_indices(&query, 4.3)?;
    let result_coords = tree.search_for_coords(&query, 4.3)?;
    assert!(tree.root.is_some());
    assert_eq!(tree.nodes.len(), 5);
    assert_eq!(result_inds.len(), 2);
    assert_eq!(result_inds, array!(2, 1));
    assert_eq!(result_coords.dim().0, 2);
    assert_eq!(result_coords.row(0), cloud.row(2));
    assert_eq!(result_coords.row(1), cloud.row(1));
    Ok(())
}
