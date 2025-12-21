use imgal::simulation::gradient::{linear_gradient_2d, linear_gradient_3d};
use imgal::statistics;
use ndarray::Array1;

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
fn statistics_max_1d() {
    let data_f64 = vec![1.0, 5.0, 3.0, 9.0, 2.0];
    let data_str = vec!["1.0", "5.0", "4.0"];
    let result_f64 = statistics::max(&data_f64);
    let result_str = statistics::max(&data_str);
    assert_eq!(result_f64, 9.0);
    assert_eq!(result_str, "5.0");

    let data_f64_array: Array1<f64> = Array1::from_vec(data_f64);
    let data_str_array: Array1<&'static str> = Array1::from_vec(data_str);
    let result_f64_array = statistics::max(&data_f64_array);
    let result_str_array = statistics::max(&data_str_array);
    assert_eq!(result_f64_array, 9.0);
    assert_eq!(result_str_array, "5.0");
}

#[test]
fn statistics_min_1d() {
    let data = vec![1.0, 5.0, 3.0, 9.0, 2.0];
    let result = statistics::min(&data);
    assert_eq!(result, 1.0);

    let data_array: Array1<f64> = Array1::from_vec(data);
    let result_array = statistics::min(&data_array);
    assert_eq!(result_array, 1.0);
}

#[test]
fn statistics_min_max_1d() {
    let data = vec![1.0, 5.0, 3.0, 9.0, 2.0];
    let (min, max) = statistics::min_max(&data);
    assert_eq!(min, 1.0);
    assert_eq!(max, 9.0);

    let data_array: Array1<f64> = Array1::from_vec(data);
    let (min_array, max_array) = statistics::min_max(&data_array);
    assert_eq!(min_array, 1.0);
    assert_eq!(max_array, 9.0);
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

#[test]
fn statistics_weighted_kendall_tau_b_perfect_positive() {
    let a = [1, 2, 3, 4, 5];
    let b = [1, 2, 3, 4, 5];
    let w = [1.0; 5];
    let tau = statistics::weighted_kendall_tau_b(&a, &b, &w).unwrap();
    assert!((tau - 1.0).abs() < 1e-12);
}

#[test]
fn statistics_weighted_kendall_tau_b_one_disagreement() {
    let a = [1, 2, 3, 4, 5];
    let b = [1, 2, 3, 5, 4];
    let w = [1.0; 5];
    let tau = statistics::weighted_kendall_tau_b(&a, &b, &w).unwrap();
    assert!((tau - 0.8).abs() < 1e-12);
}

#[test]
fn statistics_weighted_kendall_tau_b_perfect_negative() {
    let a = [1, 2, 3, 4, 5];
    let b = [5, 4, 3, 2, 1];
    let w = [1.0; 5];
    let tau = statistics::weighted_kendall_tau_b(&a, &b, &w).unwrap();
    assert!((tau + 1.0).abs() < 1e-12);
}

#[test]
fn statistics_weighted_kendall_tau_b_all_ties_returns_nan() {
    let a = [2, 2, 2, 2];
    let b = [3, 3, 3, 3];
    let w = [1.0; 4];
    let tau = statistics::weighted_kendall_tau_b(&a, &b, &w).unwrap();

    assert!(tau.is_nan());
}
