use imgal::simulation::gradient::linear_gradient_2d;
use imgal::threshold;

const OFFSET: usize = 5;
const SCALE: f64 = 20.0;
const SHAPE: (usize, usize) = (20, 20);

#[test]
fn threshold_manual_mask() {
    // create sample data and apply a manual thresohld
    let data = linear_gradient_2d(OFFSET, SCALE, SHAPE);
    let mask = threshold::manual_mask(&data, 140.0);

    // check points along the threshold boundray
    assert_eq!(data[[13, 0]], 160.0);
    assert_eq!(mask[[11, 0]], false);
    assert_eq!(mask[[13, 0]], true);
}

#[test]
fn threshold_otsu_mask() {
    let data = linear_gradient_2d(OFFSET, SCALE, SHAPE);
    let mask = threshold::otsu_mask(&data, None);

    // check points along the threshold boundary
    assert_eq!(mask[[10, 0]], false);
    assert_eq!(mask[[11, 0]], true);
}

#[test]
fn threshold_otsu_value() {
    let data = linear_gradient_2d(OFFSET, SCALE, SHAPE);

    // check if Otsu threshold value matches expected
    assert_eq!(threshold::otsu_value(&data, None), 118.671875);
}
