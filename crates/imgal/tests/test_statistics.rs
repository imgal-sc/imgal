use ndarray::arr2;

use imgal::prelude::*;
use imgal::simulation::blob::gaussian_metaballs;
use imgal::statistics::{
    effective_sample_size, kahan_sum, linear_percentile, max, min, min_max, sum,
    weighted_kendall_tau_b, weighted_merge_sort_mut,
};

const TOLERANCE: f64 = 1e-10;
const CENTER: [[f64; 2]; 1] = [[25.0, 25.0]];
const RADIUS: [f64; 1] = [20.0];
const INTENSITY: [f64; 1] = [10.0];
const FALLOFF: [f64; 1] = [2.0];
const BACKGROUND: f64 = 0.0;
const SHAPE: [usize; 2] = [50, 50];
const THREADS: Option<usize> = Some(0);

fn approx_equal(a: f64, b: f64, tol: Option<f64>) -> bool {
    (a - b).abs() < tol.unwrap_or(TOLERANCE)
}

/// Tests that `effective_sample_size` returns the expected results for data
/// that is dominated by a single weight, partially zero, uniform and all zeros.
#[test]
fn statistics_effective_sample_size_expected_results() {
    let dominant_w: [f64; 5] = [0.99, 0.001, 0.001, 0.001, 0.001];
    let part_zero_w: [f64; 5] = [1.0, 2.0, 0.0, 0.0, 0.0];
    let uniform_w: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 1.0];
    let zero_w: [f64; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];
    assert!((approx_equal(effective_sample_size(&dominant_w), 1.0080930187, None)));
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
        None,
    )?;
    let axis_par = linear_percentile(&data, 99.8, Some(0), None, THREADS)?;
    let axis_seq = linear_percentile(&data, 99.8, Some(0), None, None)?;
    let mm_par = min_max(&axis_par, None)?;
    let mm_seq = min_max(&axis_seq, None)?;
    let flat_par = linear_percentile(&data, 99.8, None, None, THREADS)?;
    let flat_seq = linear_percentile(&data, 99.8, None, None, None)?;
    assert_eq!(axis_par.shape(), [50,]);
    assert_eq!(axis_seq.shape(), [50,]);
    assert!(approx_equal(mm_par.0, 4.5777731222, None));
    assert!(approx_equal(mm_seq.0, 4.5777731222, None));
    assert!(approx_equal(mm_par.1, 9.9987757653, None));
    assert!(approx_equal(mm_seq.1, 9.9987757653, None));
    assert!(approx_equal(flat_par[0], 9.9750561771, None));
    assert!(approx_equal(flat_seq[0], 9.9750561771, None));
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
        None,
    )?;
    let i32_data: [i32; 10] = [1, 5, 3, 9, 2, 3, 0, 4, 6, 15];
    let f64_data: [f64; 10] = [1.0, 5.0, 3.0, 9.0, 2.0, 3.0, 0.0, 4.0, 6.0, 15.0];
    let str_data: [&str; 8] = ["1.0", "5.0", "3.0", "4.0", "15.0", "9.0", "0.0", "8.0"];
    assert_eq!(max(&i32_data, THREADS)?, 15);
    assert_eq!(max(&i32_data, None)?, 15);
    assert_eq!(max(&f64_data, THREADS)?, 15.0);
    assert_eq!(max(&f64_data, None)?, 15.0);
    assert_eq!(max(&str_data, THREADS)?, "9.0");
    assert_eq!(max(&str_data, None)?, "9.0");
    assert_eq!(max(&image_data, THREADS)?, 10.0);
    assert_eq!(max(&image_data, None)?, 10.0);
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
        None,
    )?;
    let i32_data: [i32; 10] = [1, 5, 3, 9, 2, 3, 0, 4, 6, 15];
    let f64_data: [f64; 10] = [1.0, 5.0, 3.0, 9.0, 2.0, 3.0, 0.0, 4.0, 6.0, 15.0];
    let str_data: [&str; 8] = ["1.0", "5.0", "3.0", "4.0", "15.0", "9.0", "0.0", "8.0"];
    assert_eq!(min(&i32_data, THREADS)?, 0);
    assert_eq!(min(&i32_data, None)?, 0);
    assert_eq!(min(&f64_data, THREADS)?, 0.0);
    assert_eq!(min(&f64_data, None)?, 0.0);
    assert_eq!(min(&str_data, THREADS)?, "0.0");
    assert_eq!(min(&str_data, None)?, "0.0");
    assert!(approx_equal(min(&image_data, THREADS)?, 2.0961138715, None));
    assert!(approx_equal(min(&image_data, None)?, 2.0961138715, None));
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
        None,
    )?;
    let i32_data: [i32; 10] = [1, 5, 3, 9, 2, 3, 0, 4, 6, 15];
    let f64_data: [f64; 10] = [1.0, 5.0, 3.0, 9.0, 2.0, 3.0, 0.0, 4.0, 6.0, 15.0];
    let str_data: [&str; 8] = ["1.0", "5.0", "3.0", "4.0", "15.0", "9.0", "0.0", "8.0"];
    assert_eq!(min_max(&i32_data, THREADS)?, (0, 15));
    assert_eq!(min_max(&i32_data, None)?, (0, 15));
    assert_eq!(min_max(&f64_data, THREADS)?, (0.0, 15.0));
    assert_eq!(min_max(&f64_data, None)?, (0.0, 15.0));
    assert_eq!(min_max(&str_data, THREADS)?, ("0.0", "9.0"));
    assert_eq!(min_max(&str_data, None)?, ("0.0", "9.0"));
    assert!(approx_equal(
        min_max(&image_data, THREADS)?.0,
        2.0961138715,
        None
    ));
    assert!(approx_equal(
        min_max(&image_data, None)?.0,
        2.0961138715,
        None
    ));
    assert!(approx_equal(min_max(&image_data, THREADS)?.1, 10.0, None));
    assert!(approx_equal(min_max(&image_data, None)?.1, 10.0, None));
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
        None,
    )?;
    let i32_data = vec![2, 5, 10, 23];
    let f64_data = vec![1.0, 10.5, 3.25, 37.11];
    let f64_error_data = vec![0.1_f64; 1000];
    assert_eq!(sum(&i32_data, THREADS), 40);
    assert_eq!(sum(&i32_data, None), 40);
    assert_eq!(sum(&f64_data, THREADS), 51.86);
    assert_eq!(sum(&f64_data, None), 51.86);
    assert!(approx_equal(sum(&f64_error_data, THREADS), 100.0, None));
    assert!(approx_equal(
        sum(&f64_error_data, None),
        99.9999999999,
        None
    ));
    assert!(approx_equal(
        sum(&image_data, THREADS),
        15630.0102099582,
        None
    ));
    assert!(approx_equal(sum(&image_data, None), 15630.0102099582, None));
    Ok(())
}

/// Tests that `weighted_kendall_tau_b` returns the expected results for perfect
/// positive correlation, perfect negative correlation, tie corretion and order
/// invariance.
#[test]
fn statistics_weighted_kendall_tau_b_expected_results() -> Result<(), ImgalError> {
    let perfect_pos_corr = ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1.0_f64; 5]);
    let perfect_neg_corr = ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1.0_f64; 5]);
    let single_diff_corr = ([1, 2, 3, 4, 5], [1, 2, 3, 5, 4], [1.0_f64; 5]);
    let all_ties_corr = ([2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [1.0_f64; 5]);
    let no_ties_fwd = (
        [10, 21, 22, 23, 30, 40, 50],
        [5, 3, 8, 6, 2, 9, 10],
        [1.0_f64; 7],
    );
    let no_ties_rev = (
        [50, 40, 30, 23, 22, 21, 10],
        [10, 9, 2, 6, 8, 3, 5],
        [1.0_f64; 7],
    );
    let with_ties_fwd = (
        [10, 20, 20, 20, 30, 40, 50],
        [50, 40, 30, 20, 20, 20, 10],
        [1.0_f64; 7],
    );
    let with_ties_rev = (
        [50, 40, 30, 20, 20, 20, 10],
        [10, 20, 20, 20, 30, 40, 50],
        [1.0_f64; 7],
    );
    assert_eq!(
        weighted_kendall_tau_b(
            &perfect_pos_corr.0,
            &perfect_pos_corr.1,
            &perfect_pos_corr.2
        )?,
        1.0
    );
    assert_eq!(
        weighted_kendall_tau_b(
            &perfect_neg_corr.0,
            &perfect_neg_corr.1,
            &perfect_neg_corr.2
        )?,
        -1.0
    );
    assert_eq!(
        weighted_kendall_tau_b(
            &single_diff_corr.0,
            &single_diff_corr.1,
            &single_diff_corr.2
        )?,
        0.8
    );
    assert!(weighted_kendall_tau_b(&all_ties_corr.0, &all_ties_corr.1, &all_ties_corr.2)?.is_nan(),);
    assert!(approx_equal(
        weighted_kendall_tau_b(&no_ties_fwd.0, &no_ties_fwd.1, &no_ties_fwd.2)?,
        0.4285714285,
        None
    ));
    assert!(approx_equal(
        weighted_kendall_tau_b(&no_ties_rev.0, &no_ties_rev.1, &no_ties_rev.2)?,
        0.4285714285,
        None
    ));
    assert!(approx_equal(
        weighted_kendall_tau_b(&with_ties_fwd.0, &with_ties_fwd.1, &with_ties_fwd.2)?,
        -0.6666666666,
        None
    ));
    assert!(approx_equal(
        weighted_kendall_tau_b(&with_ties_rev.0, &with_ties_rev.1, &with_ties_rev.2)?,
        -0.6666666666,
        None
    ));
    Ok(())
}

/// Tests that `weighted_merge_sort_mut` returns the expected weighted inversion
/// count and correctly orders the data and weights arrays. The "ping pong" data
/// tests the ping pong buffer used in the function. The short ping pong data
/// avoids a final copy while the long requires it.
#[test]
fn statistics_weighted_merge_sort_mut_expected_results() -> Result<(), ImgalError> {
    let mut simple_data = (
        [3, 10, 87, 22, 5, 15, 36, 8, 54, 1],
        [0.51, 12.83, 4.24, 9.25, 0.32, 3.22, 1.97, 0.72, 4.10, 10.7],
    );
    let mut ping_pong_short = ([8, 3, 1, 7], [1.0; 4]);
    let mut ping_pong_long = (
        [64, 34, 25, 12, 22, 11, 90, 45],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let simple_swaps = weighted_merge_sort_mut(&mut simple_data.0, &mut simple_data.1)?;
    let pp_short_swaps = weighted_merge_sort_mut(&mut ping_pong_short.0, &mut ping_pong_short.1)?;
    let pp_long_swaps = weighted_merge_sort_mut(&mut ping_pong_long.0, &mut ping_pong_long.1)?;
    assert_eq!(simple_data.0, [1, 3, 5, 8, 10, 15, 22, 36, 54, 87]);
    assert_eq!(
        simple_data.1,
        [10.7, 0.51, 0.32, 0.72, 12.83, 3.22, 9.25, 1.97, 4.1, 4.24]
    );
    assert_eq!(ping_pong_short.0, [1, 3, 7, 8]);
    assert_eq!(ping_pong_short.1, [1.0; 4]);
    assert_eq!(ping_pong_long.0, [11, 12, 22, 25, 34, 45, 64, 90]);
    assert_eq!(ping_pong_long.1, [6.0, 4.0, 5.0, 3.0, 2.0, 8.0, 1.0, 7.0]);
    assert_eq!(simple_swaps, 537.1162);
    assert_eq!(pp_short_swaps, 4.0);
    assert_eq!(pp_long_swaps, 219.0);
    Ok(())
}
