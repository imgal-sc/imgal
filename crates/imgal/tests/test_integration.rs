use imgal::distribution::normalized_gaussian;
use imgal::integration;

// helper functions
fn get_gaussian_distribution(bins: usize) -> Vec<f64> {
    normalized_gaussian(2.0, bins, 4.0, 2.0)
}

#[test]
fn integration_composite_simpson() {
    let gauss_arr = get_gaussian_distribution(512);

    // check if the function produces the expected results
    assert_eq!(
        integration::composite_simpson(&gauss_arr, None),
        0.9986155934120933
    );
}

#[test]
fn integration_midpoint() {
    let gauss_arr = get_gaussian_distribution(512);

    // check if the function produces the expected results
    assert_eq!(integration::midpoint(&gauss_arr, None), 1.0000000000000009);
}

#[test]
fn integration_simpson() {
    let gauss_arr = get_gaussian_distribution(511);

    // check if the function produces the expected results
    assert_eq!(
        integration::simpson(&gauss_arr, None).unwrap(),
        0.9986128844345734
    );
}
