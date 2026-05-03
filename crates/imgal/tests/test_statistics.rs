use imgal::error::ImgalError;
use imgal::simulation::blob::gaussian_metaballs;
use imgal::statistics::{
    effective_sample_size, kahan_sum, linear_percentile, max, min, min_max, sum,
    weighted_kendall_tau_b_correlation,
};
use ndarray::arr2;

const TOLERANCE: f64 = 1e-10;
const CENTER: [[f64; 2]; 1] = [[25.0, 25.0]];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const SHAPE: [usize; 2] = [50, 50];

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Tests that `effective_sample_size` returns the expected results data that
/// is dominated by a single weight, partially zero, uniform and all zeros.
#[test]
fn statistics_effective_sample_size_expected_results() {
    let dominant_w: [f64; 5] = [0.99, 0.001, 0.001, 0.001, 0.001];
    let part_zero_w: [f64; 5] = [1.0, 2.0, 0.0, 0.0, 0.0];
    let uniform_w: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 1.0];
    let zero_w: [f64; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];
    assert!((approx_equal(effective_sample_size(&dominant_w), 1.0080930187)));
    assert_eq!(effective_sample_size(&part_zero_w), 1.8);
    assert_eq!(effective_sample_size(&uniform_w), 5.0);
    assert_eq!(effective_sample_size(&zero_w), 0.0);
}

/// Tests that `kahan_sum` returns the expected compensated sum results for
/// integer and floating point data.
#[test]
fn statistics_kahan_sum_expected_results() -> Result<(), ImgalError> {
    let i32_data = vec![2, 5, 10, 23];
    let f64_data = vec![1.0, 10.5, 3.25, 37.11];
    let f64_error_data = vec![0.1_f64; 1000];
    let mut large_small_data = vec![1e-7_f64; 1_000_000];
    large_small_data.insert(0, 1_000_000.0);
    assert_eq!(kahan_sum(&i32_data)?, 40);
    assert_eq!(kahan_sum(&f64_data)?, 51.86);
    assert_eq!(kahan_sum(&f64_error_data)?, 100.0);
    assert_eq!(kahan_sum(&large_small_data)?, 1_000_000.1);
    Ok(())
}

/// Tests that `linear_percentile` returns the expected results for flat and
/// axis compute.
#[test]
fn statistics_linear_percentile_expected_results() -> Result<(), ImgalError> {
    let data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
    )?;
    let axis_par = linear_percentile(&data, 99.8, Some(0), None, true)?;
    let axis_seq = linear_percentile(&data, 99.8, Some(0), None, false)?;
    let mm_par = min_max(&axis_par, false)?;
    let mm_seq = min_max(&axis_seq, false)?;
    let flat_par = linear_percentile(&data, 99.8, None, None, true)?;
    let flat_seq = linear_percentile(&data, 99.8, None, None, false)?;
    assert_eq!(axis_par.shape(), [50,]);
    assert_eq!(axis_seq.shape(), [50,]);
    assert!(approx_equal(mm_par.0, 4.5777731222));
    assert!(approx_equal(mm_seq.0, 4.5777731222));
    assert!(approx_equal(mm_par.1, 9.9987757653));
    assert!(approx_equal(mm_seq.1, 9.9987757653));
    assert!(approx_equal(flat_par[0], 9.9750561771));
    assert!(approx_equal(flat_seq[0], 9.9750561771));
    Ok(())
}

/// Tests that `max` returns the maximum value from integer, floating point,
/// string arrays and images.
#[test]
fn statistics_max_expected_results() -> Result<(), ImgalError> {
    let image_data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
    )?;
    let i32_data: [i32; 10] = [1, 5, 3, 9, 2, 3, 0, 4, 6, 15];
    let f64_data: [f64; 10] = [1.0, 5.0, 3.0, 9.0, 2.0, 3.0, 0.0, 4.0, 6.0, 15.0];
    let str_data: [&str; 8] = ["1.0", "5.0", "3.0", "4.0", "15.0", "9.0", "0.0", "8.0"];
    assert_eq!(max(&i32_data, true)?, 15);
    assert_eq!(max(&i32_data, false)?, 15);
    assert_eq!(max(&f64_data, true)?, 15.0);
    assert_eq!(max(&f64_data, false)?, 15.0);
    assert_eq!(max(&str_data, true)?, "9.0");
    assert_eq!(max(&str_data, false)?, "9.0");
    assert_eq!(max(&image_data, true)?, 10.0);
    assert_eq!(max(&image_data, false)?, 10.0);
    Ok(())
}

/// Tests that `min` returns the minimum value from integer, floating point,
/// string arrays and images.
#[test]
fn statistics_min_expected_results() -> Result<(), ImgalError> {
    let image_data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
    )?;
    let i32_data: [i32; 10] = [1, 5, 3, 9, 2, 3, 0, 4, 6, 15];
    let f64_data: [f64; 10] = [1.0, 5.0, 3.0, 9.0, 2.0, 3.0, 0.0, 4.0, 6.0, 15.0];
    let str_data: [&str; 8] = ["1.0", "5.0", "3.0", "4.0", "15.0", "9.0", "0.0", "8.0"];
    assert_eq!(min(&i32_data, true)?, 0);
    assert_eq!(min(&i32_data, false)?, 0);
    assert_eq!(min(&f64_data, true)?, 0.0);
    assert_eq!(min(&f64_data, false)?, 0.0);
    assert_eq!(min(&str_data, true)?, "0.0");
    assert_eq!(min(&str_data, false)?, "0.0");
    assert!(approx_equal(min(&image_data, true)?, 2.0961138715));
    assert!(approx_equal(min(&image_data, false)?, 2.0961138715));
    Ok(())
}

/// Tests that `min_max` returns the minimum and maximum values from integer,
/// floating ppoint, string arrays and images.
#[test]
fn statistics_min_max_expected_results() -> Result<(), ImgalError> {
    let image_data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
    )?;
    let i32_data: [i32; 10] = [1, 5, 3, 9, 2, 3, 0, 4, 6, 15];
    let f64_data: [f64; 10] = [1.0, 5.0, 3.0, 9.0, 2.0, 3.0, 0.0, 4.0, 6.0, 15.0];
    let str_data: [&str; 8] = ["1.0", "5.0", "3.0", "4.0", "15.0", "9.0", "0.0", "8.0"];
    assert_eq!(min_max(&i32_data, true)?, (0, 15));
    assert_eq!(min_max(&i32_data, false)?, (0, 15));
    assert_eq!(min_max(&f64_data, true)?, (0.0, 15.0));
    assert_eq!(min_max(&f64_data, false)?, (0.0, 15.0));
    assert_eq!(min_max(&str_data, true)?, ("0.0", "9.0"));
    assert_eq!(min_max(&str_data, false)?, ("0.0", "9.0"));
    assert!(approx_equal(min_max(&image_data, true)?.0, 2.0961138715));
    assert!(approx_equal(min_max(&image_data, false)?.0, 2.0961138715));
    assert!(approx_equal(min_max(&image_data, true)?.1, 10.0));
    assert!(approx_equal(min_max(&image_data, false)?.1, 10.0));
    Ok(())
}

/// Tests that `sum` returns expected sum from integer and floating point arrays
/// as well as images.
#[test]
fn statistics_sum_expected_results() -> Result<(), ImgalError> {
    let image_data = gaussian_metaballs(
        &arr2(&CENTER),
        &RADIUS,
        &INTENSITY,
        &FALLOFF,
        BACKGROUND,
        &SHAPE,
    )?;
    let i32_data = vec![2, 5, 10, 23];
    let f64_data = vec![1.0, 10.5, 3.25, 37.11];
    let f64_error_data = vec![0.1_f64; 1000];
    let mut large_small_data = vec![1e-7_f64; 1_000_000];
    large_small_data.insert(0, 1_000_000.0);
    assert_eq!(sum(&i32_data, true), 40);
    assert_eq!(sum(&i32_data, false), 40);
    assert_eq!(sum(&f64_data, true), 51.86);
    assert_eq!(sum(&f64_data, false), 51.86);
    assert!(approx_equal(sum(&f64_error_data, true), 100.0));
    assert!(approx_equal(sum(&f64_error_data, false), 99.9999999999));
    assert!(approx_equal(
        sum(&large_small_data, true),
        1_000_000.1000000238
    ));
    assert!(approx_equal(
        sum(&large_small_data, false),
        1_000_000.1000007614
    ));
    assert!(approx_equal(sum(&image_data, true), 15630.0102099582));
    assert!(approx_equal(sum(&image_data, false), 15630.0102099582));
    Ok(())
}


// #[test]
// fn statistics_weighted_merge_sort_mut() {
//     // create data and associated weights
//     let mut d: [i32; 5] = [3, 10, 87, 22, 5];
//     let mut w: [f64; 5] = [0.51, 12.83, 4.24, 9.25, 0.32];

//     // sort the data and weights, get inversion count
//     let s = statistics::weighted_merge_sort_mut(&mut d, &mut w).unwrap();

//     // check arrays are sorted
//     assert_eq!(d, [3, 5, 10, 22, 87]);
//     assert_eq!(w, [0.51, 0.32, 12.83, 9.25, 4.24]);
//     assert_eq!(s, 47.64239999999998);
// }

// #[test]
// fn statistics_weighted_merge_sort_mut_len_4() {
//     // Note that this test and the test below ensure correct functioning of the
//     // ping-pong buffer logic. This test uses an array length where the sorted output
//     // is in the original buffer at the end of sorting, avoiding a final copy.
//     let mut d = [8, 3, 1, 7];
//     let mut w = [1.0, 1.0, 1.0, 1.0];
//     let _s = statistics::weighted_merge_sort_mut(&mut d, &mut w).unwrap();
//     assert_eq!(d, [1, 3, 7, 8]);
//     assert_eq!(w, [1.0, 1.0, 1.0, 1.0]);
// }

// #[test]
// fn statistics_weighted_merge_sort_mut_len_8() {
//     // Note that this test and the test above ensure correct functioning of the
//     // ping-pong buffer logic. This test uses an array length where the sorted output
//     // is in the internal buffer at the end of sorting, requiring a final copy.
//     let mut d = [64, 34, 25, 12, 22, 11, 90, 45];
//     let mut w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//     let _s = statistics::weighted_merge_sort_mut(&mut d, &mut w).unwrap();
//     assert_eq!(d, [11, 12, 22, 25, 34, 45, 64, 90]);
//     assert_eq!(w, [6.0, 4.0, 5.0, 3.0, 2.0, 8.0, 1.0, 7.0]);
// }

// #[test]
// fn statistics_weighted_kendall_tau_b_correlation_perfect_positive() {
//     let a = [1, 2, 3, 4, 5];
//     let b = [1, 2, 3, 4, 5];
//     let w = [1.0; 5];
//     let tau = statistics::weighted_kendall_tau_b_correlation(&a, &b, &w).unwrap();
//     assert!((tau - 1.0).abs() < 1e-12);
// }

// #[test]
// fn statistics_weighted_kendall_tau_b_correlation_one_disagreement() {
//     let a = [1, 2, 3, 4, 5];
//     let b = [1, 2, 3, 5, 4];
//     let w = [1.0; 5];
//     let tau = statistics::weighted_kendall_tau_b_correlation(&a, &b, &w).unwrap();
//     assert!((tau - 0.8).abs() < 1e-12);
// }

// #[test]
// fn statistics_weighted_kendall_tau_b_correlation_perfect_negative() {
//     let a = [1, 2, 3, 4, 5];
//     let b = [5, 4, 3, 2, 1];
//     let w = [1.0; 5];
//     let tau = statistics::weighted_kendall_tau_b_correlation(&a, &b, &w).unwrap();
//     assert!((tau + 1.0).abs() < 1e-12);
// }

// #[test]
// fn statistics_weighted_kendall_tau_b_correlation_all_ties_returns_nan() {
//     let a = [2, 2, 2, 2];
//     let b = [3, 3, 3, 3];
//     let w = [1.0; 4];
//     let tau = statistics::weighted_kendall_tau_b_correlation(&a, &b, &w).unwrap();

//     assert!(tau.is_nan());
// }

// #[test]
// fn statistics_weighted_kendall_tau_b_correlation_order_invariant() {
//     let a_no_ties_fwd = [10, 21, 22, 23, 30, 40, 50];
//     let a_no_ties_rev = [50, 40, 30, 23, 22, 21, 10];
//     let a_with_ties_fwd = [10, 20, 20, 20, 30, 40, 50];
//     let a_with_ties_rev = [50, 40, 30, 20, 20, 20, 10];
//     let b_no_ties_fwd = [5, 3, 8, 6, 2, 9, 10];
//     let b_no_ties_rev = [10, 9, 2, 6, 8, 3, 5];
//     let w = [1.0; 7];
//     let tau_no_ties_fwd =
//         statistics::weighted_kendall_tau_b_correlation(&a_no_ties_fwd, &b_no_ties_fwd, &w).unwrap();
//     let tau_no_ties_rev =
//         statistics::weighted_kendall_tau_b_correlation(&a_no_ties_rev, &b_no_ties_rev, &w).unwrap();
//     let tau_with_ties_fwd =
//         statistics::weighted_kendall_tau_b_correlation(&a_with_ties_fwd, &b_no_ties_fwd, &w)
//             .unwrap();
//     let tau_with_ties_rev =
//         statistics::weighted_kendall_tau_b_correlation(&a_with_ties_rev, &b_no_ties_rev, &w)
//             .unwrap();
//     assert_eq!(tau_no_ties_fwd, 0.42857142857142855);
//     assert_eq!(tau_no_ties_rev, 0.42857142857142855);
//     assert_eq!(tau_with_ties_fwd, 0.41147559989891175);
//     assert_eq!(tau_with_ties_rev, 0.41147559989891175);
// }
