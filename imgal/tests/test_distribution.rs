use imgal::distribution;
use imgal::integration::midpoint;

#[test]
fn distribution_inverse_normal_cdf() {
    // store undefined result for assert later
    let undefined_result = distribution::inverse_normal_cdf(0.0).unwrap();

    // check if the function produces the expected results
    assert_eq!(
        distribution::inverse_normal_cdf(0.975).unwrap(),
        1.959963986120195
    );
    assert_eq!(
        distribution::inverse_normal_cdf(0.1).unwrap(),
        -1.2815515641401563
    );
    assert_eq!(distribution::inverse_normal_cdf(0.5).unwrap(), 0.0);
    assert!(undefined_result.is_nan());
}

#[test]
fn distribution_normalized_gaussian() {
    // create a gaussian distribution
    let gauss_arr = distribution::normalized_gaussian(2.0, 256, 4.0, 2.0);

    // check if a point on the curve is as expected and its integral is ~1.0
    assert_eq!(gauss_arr[100], 0.004465507286912305);
    assert_eq!(midpoint(&gauss_arr, None), 1.0000000000000007);
}
