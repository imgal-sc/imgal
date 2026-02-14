use imgal::distribution::normalized_gaussian;
use imgal::integration;

// helper functions
fn ensure_within_tolerance(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

fn get_gaussian_distribution(bins: usize) -> Vec<f64> {
    normalized_gaussian(2.0, bins, 4.0, 2.0, false)
}

#[test]
fn integration_composite_simpson() {
    let gauss_arr = get_gaussian_distribution(512);
    let result_par = integration::composite_simpson(&gauss_arr, None, true);
    let result_seq = integration::composite_simpson(&gauss_arr, None, false);

    // check if the function produces the expected results
    assert!(ensure_within_tolerance(
        result_par,
        0.9986155934120933,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        result_seq,
        0.9986155934120933,
        1e-12
    ));
}

#[test]
fn integration_midpoint() {
    let gauss_arr = get_gaussian_distribution(512);
    let result_par = integration::midpoint(&gauss_arr, None, true);
    let result_seq = integration::midpoint(&gauss_arr, None, false);

    // check if the function produces the expected results
    assert!(ensure_within_tolerance(result_par, 1.0, 1e-12));
    assert!(ensure_within_tolerance(result_seq, 1.0, 1e-12));
}

#[test]
fn integration_simpson() {
    let gauss_arr = get_gaussian_distribution(511);
    let result_par = integration::simpson(&gauss_arr, None, true).unwrap();
    let result_seq = integration::simpson(&gauss_arr, None, false).unwrap();

    // check if the function produces the expected results
    assert!(ensure_within_tolerance(
        result_par,
        0.9986128844345734,
        1e-12
    ));
    assert!(ensure_within_tolerance(
        result_seq,
        0.9986128844345734,
        1e-12
    ));
}
