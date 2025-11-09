use imgal::statistics;
use ndarray::{Array1};

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
fn statistics_max_1d() {
    let data = vec![1.0, 5.0, 3.0, 9.0, 2.0];
    let result = statistics::max(&data);
    assert_eq!(result, 9.0);

    let data_array: Array1<f64> = Array1::from_vec(data);
    let result_array = statistics::max(&data_array);
    assert_eq!(result_array, 9.0);
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