use ndarray::array;

use imgal::error::ImgalError;
use imgal::spatial::KDTree;

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
