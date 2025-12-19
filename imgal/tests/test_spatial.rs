use ndarray::array;

use imgal::spatial::KDTree;

#[test]
fn sptial_kd_tree() {
    // a 3D point cloud
    let cloud = array![
        [-2.7, 3.9, 5.0],
        [0.1, 0.0, 4.0],
        [1.4, 0.2, 2.1],
        [-3.2, -1.8, -2.3],
        [-4.9, -3.7, -1.1],
    ];

    // query the origin and find points near it
    let tree = KDTree::build(&cloud);
    let query = [0.0, 0.0, 0.0];
    let result_inds = tree.search_for_indices(&query, 4.3).unwrap();
    let result_coords = tree.search_for_coords(&query, 4.3).unwrap();

    // check that the has the expected nodes
    assert!(tree.root.is_some());
    assert_eq!(tree.nodes.len(), 5);

    // check the number of points and the indices
    assert_eq!(result_inds.len(), 2);
    assert_eq!(result_inds, [2, 1]);

    // check the number of points and the coordinates
    assert_eq!(result_coords.dim().0, 2);
    assert_eq!(result_coords.row(0), cloud.row(2));
    assert_eq!(result_coords.row(1), cloud.row(1));
}
