use ndarray::{Array1, arr2, array, s};

use imgal::ImgalError;
use imgal::spatial::KDTree;
use imgal::spatial::convex_hull::{chan_2d, graham_scan, jarvis_march, quickhull_3d};
use imgal::spatial::geometry::{orient_pred_3d, tetrahedron_volume};
use imgal::spatial::halfplane::{
    edge_to_halfplane, halfplane_intersection, hull_to_halfplane, inside_halfplane_interior,
};
use imgal::spatial::halfspace::{
    face_to_halfspace, halfspace_intersection, hull_to_halfspace, inside_halfspace_interior,
};

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
const HALFPLANES: [[f64; 3]; 4] = [
    [-3.5, 1.6, -9.81],
    [1.4, 4.5, -19.72],
    [5.1, -1.4, -14.9],
    [-3.0, -4.7, -1.73],
];
const HALFPLANE_VERTS: [[f64; 2]; 4] = [[-2.3, 1.1], [-0.7, 4.6], [3.8, 3.2], [2.4, -1.9]];
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64, tol: Option<f64>) -> bool {
    (a - b).abs() < tol.unwrap_or(TOLERANCE)
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

/// Tests that `orient_pred_3d` returns the expected orientation of the input
/// tetrahedron as defined by vertex `d`.
#[test]
fn geometry_orient_pred_3d_expected_results() -> Result<(), ImgalError> {
    let a = arr1(&[3.2, 0.4, 8.5]);
    let b = arr1(&[6.7, 1.1, 9.8]);
    let c = arr1(&[0.0, 4.9, 5.1]);
    let d_above = arr1(&[0.0, 1.2, 8.0]);
    let d_below = arr1(&[0.0, 1.2, -8.0]);
    let d_coplanar = arr1(&[3.2, 0.4, 8.5]);
    assert!(orient_pred_3d(&a, &b, &c, &d_above)?.is_sign_positive());
    assert!(orient_pred_3d(&a, &b, &c, &d_below)?.is_sign_negative());
    assert_eq!(orient_pred_3d(&a, &b, &c, &d_coplanar)?, 0.0);
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
        -2.6666666666,
        None
    ));
    assert!(approx_equal(
        tetrahedron_volume(
            pos_vol_tet.row(0),
            pos_vol_tet.row(1),
            pos_vol_tet.row(2),
            pos_vol_tet.row(3)
        )?,
        2.6666666666,
        None
    ));
    Ok(())
}

/// Tests that `edge_to_halfplane` returns the expected outward-facing line
/// equation for an edge.
#[test]
fn halfplane_edge_to_halfplane_expected_results() -> Result<(), ImgalError> {
    let a = array![HALFPLANE_VERTS[0][0], HALFPLANE_VERTS[0][1]];
    let b = array![HALFPLANE_VERTS[1][0], HALFPLANE_VERTS[1][1]];
    let halfplane = edge_to_halfplane(&a, &b)?;
    assert!(
        halfplane
            .iter()
            .zip(HALFPLANES[0])
            .all(|(&actual, expected)| approx_equal(actual, expected, None))
    );
    Ok(())
}

/// Tests that `halfplane_intersection` reconstructs an asymmetric quadrilateral
/// from its halfplane representation.
#[test]
fn halfplane_halfplane_intersection_expected_results() -> Result<(), ImgalError> {
    let interior = [0.5, 1.0];
    let num_verts = HALFPLANE_VERTS.len();
    let verts_par = halfplane_intersection(&HALFPLANES, &interior, THREADS)?;
    let verts_seq = halfplane_intersection(&HALFPLANES, &interior, None)?;
    assert_eq!(verts_par.dim().0, num_verts);
    assert_eq!(verts_seq.dim().0, num_verts);
    Ok(())
}

/// Tests that `hull_to_halfplane` returns the expected halfplane vectors for
/// each edge of an asymmetric quadrilateral.
#[test]
fn halfplane_hull_to_halfplane_expected_results() -> Result<(), ImgalError> {
    let hp_expected = arr2(&HALFPLANES);
    let hp_par = hull_to_halfplane(&HALFPLANE_VERTS, THREADS)?;
    let hp_seq = hull_to_halfplane(&HALFPLANE_VERTS, None)?;
    assert_eq!(hp_par.dim(), (4, 3));
    assert_eq!(hp_seq.dim(), (4, 3));
    assert!(
        hp_seq
            .iter()
            .zip(hp_expected.iter())
            .all(|(&actual, &expected)| approx_equal(actual, expected, None))
    );
    Ok(())
}

/// Tests that `inside_halfplane_interior` returns the expected results for
/// points that are inside, outside, and on the boundary of an asymmetric
/// quadrilateral.
#[test]
fn halfplane_inside_halfplane_interior_expected_results() -> Result<(), ImgalError> {
    let verts = arr2(&HALFPLANE_VERTS);
    let hp = hull_to_halfplane(&verts, None)?;
    let inside = array![0.5, 1.0];
    let outside = array![2.5, 4.0];
    let boundary = array![0.0, -hp[[1, 2]] / hp[[1, 1]]];
    assert!(inside_halfplane_interior(
        &hp,
        &inside,
        false,
        THREADS
    )?);
    assert!(inside_halfplane_interior(
        &hp,
        &inside,
        false,
        None
    )?);
    assert!(inside_halfplane_interior(
        &hp,
        &inside,
        true,
        THREADS
    )?);
    assert!(inside_halfplane_interior(&hp, &inside, true, None)?);
    assert!(!inside_halfplane_interior(
        &hp,
        &outside,
        false,
        THREADS
    )?);
    assert!(!inside_halfplane_interior(
        &hp,
        &outside,
        false,
        None
    )?);
    assert!(!inside_halfplane_interior(
        &hp,
        &outside,
        true,
        THREADS
    )?);
    assert!(!inside_halfplane_interior(
        &hp,
        &outside,
        true,
        None
    )?);
    assert!(inside_halfplane_interior(
        &hp,
        &boundary,
        true,
        THREADS
    )?);
    assert!(inside_halfplane_interior(
        &hp,
        &boundary,
        true,
        None
    )?);
    assert!(!inside_halfplane_interior(
        &hp,
        &boundary,
        false,
        THREADS
    )?);
    assert!(!inside_halfplane_interior(
        &hp,
        &boundary,
        false,
        None
    )?);
    Ok(())
}

/// Tests that `face_to_halfspace` returns the expected halfspace normal vector
/// values.
#[test]
fn halfspace_face_to_halfspace_expected_results() -> Result<(), ImgalError> {
    let a_ideal = array![1.0, 2.0, 3.0];
    let b_ideal = array![4.0, 0.0, 1.0];
    let c_ideal = array![0.0, 3.0, 5.0];
    let a_degen = array![0.0, 0.0, 0.0];
    let b_degen = array![0.0, 0.0, 1.0];
    let c_degen = array![0.0, 1.0, 0.0];
    let hs_ideal = face_to_halfspace(&a_ideal, &b_ideal, &c_ideal)?;
    let hs_degen = face_to_halfspace(&a_degen, &b_degen, &c_degen)?;
    assert_eq!(hs_ideal, Array1::from_vec(vec![-2.0, -4.0, 1.0, 7.0]));
    assert_eq!(hs_degen, Array1::from_vec(vec![-1.0, 0.0, 0.0, -0.0]));
    Ok(())
}

/// Tests that `halfspace_intersection` returns the expected number of vertices
/// and faces for a regular octahedron.
#[test]
fn halfspace_halfspace_intersection_expected_results() -> Result<(), ImgalError> {
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
    let (oct_verts_par, oct_faces_par) = halfspace_intersection(&oct_hs, &oct_interior, THREADS)?;
    let (oct_verts_seq, oct_faces_seq) = halfspace_intersection(&oct_hs, &oct_interior, None)?;
    assert_eq!(oct_verts_par.dim().0, 6);
    assert_eq!(oct_verts_seq.dim().0, 6);
    assert_eq!(oct_faces_par.dim().0, 8);
    assert_eq!(oct_faces_seq.dim().0, 8);
    Ok(())
}

/// Tests that `hull_to_halfspace` returns the expected halfspace vectors for
/// each face of an axis-aligned tetrahedron.
#[test]
fn halfspace_hull_to_halfspace_expected_results() -> Result<(), ImgalError> {
    let vertices = arr2(&[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);
    let faces = arr2(&[[0, 2, 1], [0, 1, 3], [0, 3, 2], [3, 1, 2]]);
    let hs_par = hull_to_halfspace(&vertices, &faces, THREADS)?;
    let hs_seq = hull_to_halfspace(&vertices, &faces, None)?;
    assert_eq!(hs_par.dim(), (4, 4));
    assert_eq!(hs_seq.dim(), (4, 4));
    assert_eq!(hs_seq.row(0), array![1.0, 0.0, 0.0, -0.0]);
    assert_eq!(hs_seq.row(1), array![0.0, 1.0, 0.0, -0.0]);
    assert_eq!(hs_seq.row(2), array![0.0, 0.0, 1.0, -0.0]);
    assert_eq!(hs_seq.row(3), array![-1.0, -1.0, -1.0, 1.0]);
    Ok(())
}

/// Tests that `inside_halfspace_interior` returns the expected results for
/// points that are inside, outside, and on the boundary of a cube.
#[test]
fn halfspace_inside_halfspace_interior_expected_results() -> Result<(), ImgalError> {
    let cube_hs = arr2(&[
        [1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, -1.0, -1.0],
    ]);
    let inside = array![0.0, 0.0, 0.0];
    let outside = array![2.0, 0.0, 0.0];
    let boundary = array![1.0, 0.0, 0.0];
    assert!(inside_halfspace_interior(
        &cube_hs, &inside, false, THREADS
    )?);
    assert!(inside_halfspace_interior(&cube_hs, &inside, false, None)?);
    assert!(inside_halfspace_interior(&cube_hs, &inside, true, THREADS)?);
    assert!(inside_halfspace_interior(&cube_hs, &inside, true, None)?);
    assert!(!inside_halfspace_interior(
        &cube_hs, &outside, false, THREADS
    )?);
    assert!(!inside_halfspace_interior(&cube_hs, &outside, false, None)?);
    assert!(!inside_halfspace_interior(
        &cube_hs, &outside, true, THREADS
    )?);
    assert!(!inside_halfspace_interior(&cube_hs, &outside, true, None)?);
    assert!(inside_halfspace_interior(
        &cube_hs, &boundary, true, THREADS
    )?);
    assert!(inside_halfspace_interior(&cube_hs, &boundary, true, None)?);
    assert!(!inside_halfspace_interior(
        &cube_hs, &boundary, false, THREADS
    )?);
    assert!(!inside_halfspace_interior(
        &cube_hs, &boundary, false, None
    )?);
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
