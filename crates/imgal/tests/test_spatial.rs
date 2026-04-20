use ndarray::{arr2, array, s};

use imgal::error::ImgalError;
use imgal::spatial::KDTree;
use imgal::spatial::convex_hull::{chan_2d, graham_scan, jarvis_march};

const POINTS: [[f64; 2]; 12] = [
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

/// Tests that `chan_2d` returns the expected convex hull with the start  point
/// at index `0` and hull size.
#[test]
fn spatial_chan_2d_expected_results() -> Result<(), ImgalError> {
    let points = arr2(&POINTS);
    let hull = chan_2d(&points, false)?;
    assert_eq!(hull.slice(s![0, ..]), array![0.4, -2.5]);
    assert_eq!(hull.dim().0, 6);
    Ok(())
}

/// Tests that `graham_scan` returns the expected convex hull with the pivot
/// point at index `0` and hull size.
#[test]
fn spatial_graham_scan_expected_results() -> Result<(), ImgalError> {
    let points = arr2(&POINTS);
    let hull = graham_scan(&points, false)?;
    assert_eq!(hull.slice(s![0, ..]), array![-11.3, 3.4]);
    assert_eq!(hull.dim().0, 6);
    Ok(())
}

/// Tests that `jarvis_march` returns the expected convex hull with the start
/// point at index `0` and hull size.
#[test]
fn spatial_jarvis_march_expected_results() -> Result<(), ImgalError> {
    let points = arr2(&POINTS);
    let hull = jarvis_march(&points, false)?;
    assert_eq!(hull.slice(s![0, ..]), array![0.4, -2.5]);
    assert_eq!(hull.dim().0, 6);
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
    assert_eq!(result_inds, [2, 1]);
    assert_eq!(result_coords.dim().0, 2);
    assert_eq!(result_coords.row(0), cloud.row(2));
    assert_eq!(result_coords.row(1), cloud.row(1));
    Ok(())
}
