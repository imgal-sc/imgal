use ndarray::Array;

use imgal::image;
use imgal::simulation::gradient::linear_gradient_2d;
use imgal::statistics::min_max;

const OFFSET: usize = 5;
const SCALE: f64 = 20.0;
const SHAPE: (usize, usize) = (20, 20);

#[test]
fn image_histogram() {
    // create sample data and get the histogram
    let data = linear_gradient_2d(OFFSET, SCALE, SHAPE);
    let hist = image::histogram(data.view().into_dyn(), Some(20));

    // wrap hist vector as an array for assert tests
    let arr = Array::from_vec(hist);
    let mm = min_max(arr.view().into_dyn());

    // check histogram min and max
    assert_eq!(mm.0, 0);
    assert_eq!(mm.1, 120);
    // check if histogram has expected values
    assert_eq!(arr[0], 120);
    assert_eq!(arr[10], 20);
    assert_eq!(arr.len(), 20);
}

#[test]
fn image_histogram_bin_range() {
    let (start, end) = image::histogram_bin_range(30, 0, 1200, 256);

    // check if bin index range has expected values
    assert_eq!(start, 140);
    assert_eq!(end, 145);
}

#[test]
fn image_histogram_bin_value() {
    // check if bin index value is as expected
    assert_eq!(image::histogram_bin_midpoint(30, 0, 1200, 256), 142);
}

#[test]
fn image_percentile_normalize() {
    // create sample data and percentile normalize
    let data = linear_gradient_2d(OFFSET, SCALE, SHAPE);
    let data_norm = image::percentile_normalize(&data, 1.0, 99.8, None, None).unwrap();

    // check if the original array has expected values
    assert_eq!(data[[19, 0]], 280.0);
    assert_eq!(data[[9, 0]], 80.0);
    // check if the percentile normalized array has expected values
    assert_eq!(data_norm[[19, 0]], 1.0);
    assert_eq!(data_norm[[9, 0]], 0.2857142857142857);
}
