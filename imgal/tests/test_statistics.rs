use imgal::simulation::gradient::{linear_gradient_2d, linear_gradient_3d};
use imgal::statistics;

#[test]
fn statistics_effective_sample_size() {
    // create test data
    let dominant_w: [f64; 5] = [0.99, 0.001, 0.001, 0.001, 0.001];
    let part_zero_w: [f64; 5] = [1.0, 2.0, 0.0, 0.0, 0.0];
    let uniform_w: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 1.0];
    let zero_w: [f64; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];

    assert_eq!(
        statistics::effective_sample_size(&dominant_w),
        1.0080930187000563
    );
    assert_eq!(statistics::effective_sample_size(&part_zero_w), 1.8);
    assert_eq!(statistics::effective_sample_size(&uniform_w), 5.0);
    assert_eq!(statistics::effective_sample_size(&zero_w), 0.0);
}

#[test]
fn statistics_linear_percentile() {
    // create data with known values
    let data_2d = linear_gradient_2d(5, 20.0, (20, 20));
    let data_3d = linear_gradient_3d(5, 20.0, (20, 20, 20));

    // compute percentiles
    let flat_2d = statistics::linear_percentile(&data_2d, 99.8, None, None).unwrap();
    let flat_3d = statistics::linear_percentile(&data_3d, 99.8, None, None).unwrap();
    let axis_2d = statistics::linear_percentile(&data_2d, 99.8, Some(0), None).unwrap();
    let axis_3d = statistics::linear_percentile(&data_3d, 99.8, Some(0), None).unwrap();

    assert_eq!(flat_2d[0], 280.0);
    assert_eq!(flat_3d[0], 280.0);
    assert_eq!(axis_2d.shape(), [20,]);
    assert_eq!(axis_2d[0], 279.23999999999995);
    assert_eq!(axis_3d.shape(), [20, 20]);
    assert_eq!(axis_3d[[0, 0]], 279.23999999999995);
}

#[test]
fn statistics_sum() {
    // create some test vecs
    let int_data = vec![2, 5, 10, 23];
    let float_data = vec![1.0, 10.5, 3.25, 37.11];

    // assert arrays
    assert_eq!(statistics::sum(&int_data), 40);
    assert_eq!(statistics::sum(&float_data), 51.86);
}

#[test]
fn statistics_weighted_merge_sort_mut() {
    // create data and associated weights
    let mut d: [i32; 5] = [3, 10, 87, 22, 5];
    let mut w: [f64; 5] = [0.51, 12.83, 4.24, 9.25, 0.32];

    // sort the data and weights, get inversion count
    let s = statistics::weighted_merge_sort_mut(&mut d, &mut w).unwrap();

    // check arrays are sorted
    assert_eq!(d, [3, 5, 10, 22, 87]);
    assert_eq!(w, [0.51, 0.32, 12.83, 9.25, 4.24]);
    assert_eq!(s, 47.64239999999998);
}
