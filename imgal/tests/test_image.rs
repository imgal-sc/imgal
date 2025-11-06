use ndarray::Array;

use imgal::image;
use imgal::simulation::gradient::linear_2d;
use imgal::statistics::min_max;

#[test]
fn image_histogram() {
    // create data with known values and get the histogram
    let data = linear_2d(5, 20.0, (20, 20));
    let hist = image::histogram(data.view().into_dyn(), Some(20));

    // wrap hist vector as an array for assert tests
    let arr = Array::from_vec(hist);
    let mm = min_max(arr.view().into_dyn());

    assert_eq!(mm.0, 0);
    assert_eq!(mm.1, 120);
    assert_eq!(arr[0], 120);
    assert_eq!(arr[10], 20);
    assert_eq!(arr.len(), 20);
}

#[test]
fn image_histogram_bin_range() {
    let (start, end) = image::histogram_bin_range(30, 0, 1200, 256);

    assert_eq!(start, 140);
    assert_eq!(end, 145);
}

#[test]
fn image_histogram_bin_value() {
    assert_eq!(image::histogram_bin_midpoint(30, 0, 1200, 256), 142);
}
