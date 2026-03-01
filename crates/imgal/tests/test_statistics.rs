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
fn statistics_kahan_sum() {
    let int_data = vec![2, 5, 10, 23];
    let float_data = vec![1.0, 10.5, 3.25, 37.11];
    assert_eq!(statistics::kahan_sum(&int_data), 40);
    assert_eq!(statistics::kahan_sum(&float_data), 51.86);
}

#[test]
fn statistics_kahan_sum_compensates_floating_point_error() {
    let data = vec![0.1f64; 1000];
    assert_eq!(statistics::kahan_sum(&data), 100.0);
}

#[test]
fn statistics_kahan_sum_large_and_small_values() {
    let mut data = vec![1e-7f64; 1_000_000];
    data.insert(0, 1_000_000.0);
    assert_eq!(statistics::kahan_sum(&data), 1_000_000.1);
}

#[test]
fn statistics_kahan_sum_empty_array() {
    let empty: Vec<f64> = vec![];
    assert_eq!(statistics::kahan_sum(&empty), 0.0);
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
    let result_f64_par = statistics::max(&data_f64, true).unwrap();
    let result_str_par = statistics::max(&data_str, true).unwrap();
    let result_f64_seq = statistics::max(&data_f64, false).unwrap();
    let result_str_seq = statistics::max(&data_str, false).unwrap();
    assert_eq!(result_f64_par, 9.0);
    assert_eq!(result_str_par, "5.0");
    assert_eq!(result_f64_seq, 9.0);
    assert_eq!(result_str_seq, "5.0");

    let data_f64_array: Array1<f64> = Array1::from_vec(data_f64);
    let data_str_array: Array1<&'static str> = Array1::from_vec(data_str);
    let result_f64_array_par = statistics::max(&data_f64_array, true).unwrap();
    let result_str_array_par = statistics::max(&data_str_array, true).unwrap();
    let result_f64_array_seq = statistics::max(&data_f64_array, false).unwrap();
    let result_str_array_seq = statistics::max(&data_str_array, false).unwrap();
    assert_eq!(result_f64_array_par, 9.0);
    assert_eq!(result_str_array_par, "5.0");
    assert_eq!(result_f64_array_seq, 9.0);
    assert_eq!(result_str_array_seq, "5.0");
}

#[test]
fn statistics_min_1d() {
    let data_f64 = vec![1.0, 5.0, 3.0, 9.0, 2.0];
    let data_str = vec!["1.0", "5.0", "4.0"];
    let result_f64_par = statistics::min(&data_f64, true).unwrap();
    let result_str_par = statistics::min(&data_str, true).unwrap();
    let result_f64_seq = statistics::min(&data_f64, false).unwrap();
    let result_str_seq = statistics::min(&data_str, false).unwrap();
    assert_eq!(result_f64_par, 1.0);
    assert_eq!(result_str_par, "1.0");
    assert_eq!(result_f64_seq, 1.0);
    assert_eq!(result_str_seq, "1.0");

    let data_f64_array: Array1<f64> = Array1::from_vec(data_f64);
    let data_str_array: Array1<&'static str> = Array1::from_vec(data_str);
    let result_f64_array_par = statistics::min(&data_f64_array, true).unwrap();
    let result_str_array_par = statistics::min(&data_str_array, true).unwrap();
    let result_f64_array_seq = statistics::min(&data_f64_array, false).unwrap();
    let result_str_array_seq = statistics::min(&data_str_array, false).unwrap();
    assert_eq!(result_f64_array_par, 1.0);
    assert_eq!(result_str_array_par, "1.0");
    assert_eq!(result_f64_array_seq, 1.0);
    assert_eq!(result_str_array_seq, "1.0");
}

#[test]
fn statistics_min_max_1d() {
    let data_f64 = vec![1.0, 5.0, 3.0, 9.0, 2.0];
    let data_str = vec!["1.0", "5.0", "4.0"];
    let result_f64_par = statistics::min_max(&data_f64, true).unwrap();
    let result_str_par = statistics::min_max(&data_str, true).unwrap();
    let result_f64_seq = statistics::min_max(&data_f64, false).unwrap();
    let result_str_seq = statistics::min_max(&data_str, false).unwrap();
    assert_eq!(result_f64_par, (1.0, 9.0));
    assert_eq!(result_str_par, ("1.0", "5.0"));
    assert_eq!(result_f64_seq, (1.0, 9.0));
    assert_eq!(result_str_seq, ("1.0", "5.0"));

    let data_f64_array: Array1<f64> = Array1::from_vec(data_f64);
    let data_str_array: Array1<&'static str> = Array1::from_vec(data_str);
    let result_f64_array_par = statistics::min_max(&data_f64_array, true).unwrap();
    let result_str_array_par = statistics::min_max(&data_str_array, true).unwrap();
    let result_f64_array_seq = statistics::min_max(&data_f64_array, false).unwrap();
    let result_str_array_seq = statistics::min_max(&data_str_array, false).unwrap();
    assert_eq!(result_f64_array_par, (1.0, 9.0));
    assert_eq!(result_str_array_par, ("1.0", "5.0"));
    assert_eq!(result_f64_array_seq, (1.0, 9.0));
    assert_eq!(result_str_array_seq, ("1.0", "5.0"));
}

#[test]
fn statistics_sum() {
    // create some test vecs
    let int_data = vec![2, 5, 10, 23];
    let float_data = vec![1.0, 10.5, 3.25, 37.11];

    // assert arrays
    assert_eq!(statistics::sum(&int_data, false), 40);
    assert_eq!(statistics::sum(&float_data, false), 51.86);
    assert_eq!(statistics::sum(&int_data, true), 40);
    assert_eq!(statistics::sum(&float_data, true), 51.86);
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
fn statistics_weighted_merge_sort_mut_len_4() {
    // Note that this test and the test below ensure correct functioning of the
    // ping-pong buffer logic. This test uses an array length where the sorted output
    // is in the original buffer at the end of sorting, avoiding a final copy.
    let mut d = [8, 3, 1, 7];
    let mut w = [1.0, 1.0, 1.0, 1.0];
    let _s = statistics::weighted_merge_sort_mut(&mut d, &mut w).unwrap();
    assert_eq!(d, [1, 3, 7, 8]);
    assert_eq!(w, [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn statistics_weighted_merge_sort_mut_len_8() {
    // Note that this test and the test above ensure correct functioning of the
    // ping-pong buffer logic. This test uses an array length where the sorted output
    // is in the internal buffer at the end of sorting, requiring a final copy.
    let mut d = [64, 34, 25, 12, 22, 11, 90, 45];
    let mut w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let _s = statistics::weighted_merge_sort_mut(&mut d, &mut w).unwrap();
    assert_eq!(d, [11, 12, 22, 25, 34, 45, 64, 90]);
    assert_eq!(w, [6.0, 4.0, 5.0, 3.0, 2.0, 8.0, 1.0, 7.0]);
}

#[test]
fn statistics_weighted_kendall_tau_b_correlation_perfect_positive() {
    let a = [1, 2, 3, 4, 5];
    let b = [1, 2, 3, 4, 5];
    let w = [1.0; 5];
    let tau = statistics::weighted_kendall_tau_b_correlation(&a, &b, &w).unwrap();
    assert!((tau - 1.0).abs() < 1e-12);
}

#[test]
fn statistics_weighted_kendall_tau_b_correlation_one_disagreement() {
    let a = [1, 2, 3, 4, 5];
    let b = [1, 2, 3, 5, 4];
    let w = [1.0; 5];
    let tau = statistics::weighted_kendall_tau_b_correlation(&a, &b, &w).unwrap();
    assert!((tau - 0.8).abs() < 1e-12);
}

#[test]
fn statistics_weighted_kendall_tau_b_correlation_perfect_negative() {
    let a = [1, 2, 3, 4, 5];
    let b = [5, 4, 3, 2, 1];
    let w = [1.0; 5];
    let tau = statistics::weighted_kendall_tau_b_correlation(&a, &b, &w).unwrap();
    assert!((tau + 1.0).abs() < 1e-12);
}

#[test]
fn statistics_weighted_kendall_tau_b_correlation_all_ties_returns_nan() {
    let a = [2, 2, 2, 2];
    let b = [3, 3, 3, 3];
    let w = [1.0; 4];
    let tau = statistics::weighted_kendall_tau_b_correlation(&a, &b, &w).unwrap();

    assert!(tau.is_nan());
}

#[test]
fn statistics_weighted_kendall_tau_b_correlation_order_invariant() {
    let a_no_ties_fwd = [10, 21, 22, 23, 30, 40, 50];
    let a_no_ties_rev = [50, 40, 30, 23, 22, 21, 10];
    let a_with_ties_fwd = [10, 20, 20, 20, 30, 40, 50];
    let a_with_ties_rev = [50, 40, 30, 20, 20, 20, 10];
    let b_no_ties_fwd = [5, 3, 8, 6, 2, 9, 10];
    let b_no_ties_rev = [10, 9, 2, 6, 8, 3, 5];
    let w = [1.0; 7];
    let tau_no_ties_fwd =
        statistics::weighted_kendall_tau_b_correlation(&a_no_ties_fwd, &b_no_ties_fwd, &w).unwrap();
    let tau_no_ties_rev =
        statistics::weighted_kendall_tau_b_correlation(&a_no_ties_rev, &b_no_ties_rev, &w).unwrap();
    let tau_with_ties_fwd =
        statistics::weighted_kendall_tau_b_correlation(&a_with_ties_fwd, &b_no_ties_fwd, &w)
            .unwrap();
    let tau_with_ties_rev =
        statistics::weighted_kendall_tau_b_correlation(&a_with_ties_rev, &b_no_ties_rev, &w)
            .unwrap();
    assert_eq!(tau_no_ties_fwd, 0.42857142857142855);
    assert_eq!(tau_no_ties_rev, 0.42857142857142855);
    assert_eq!(tau_with_ties_fwd, 0.41147559989891175);
    assert_eq!(tau_with_ties_rev, 0.41147559989891175);
}
