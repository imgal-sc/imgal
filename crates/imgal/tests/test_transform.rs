use imgal::simulation::gradient::{linear_gradient_2d, linear_gradient_3d};
use imgal::transform::pad;

const PAD_CONFIG_2D: [usize; 2] = [5, 5];
const PAD_CONFIG_3D: [usize; 3] = [5, 5, 5];

#[test]
fn pad_constant_pad() {
    // create image data
    let data_2d = linear_gradient_2d(1, 100.0, (20, 20));
    let data_3d = linear_gradient_3d(1, 100.0, (20, 20, 20));

    // pad test images isometrically with a constant value
    let pad_2d = pad::constant_pad(&data_2d, 900.0, &PAD_CONFIG_2D, None).unwrap();
    let pad_3d = pad::constant_pad(&data_3d, 900.0, &PAD_CONFIG_3D, None).unwrap();

    // assert padded shape and contents are copied
    assert_eq!(pad_2d.shape(), &[30, 30]);
    assert_eq!(pad_3d.shape(), &[30, 30, 30]);
    assert_eq!(pad_2d[[2, 2]], 900.0);
    assert_eq!(pad_3d[[2, 2, 2]], 900.0);
    assert_eq!(pad_2d[[24, 24]], 1800.0);
    assert_eq!(pad_3d[[24, 24, 24]], 1800.0);
}

#[test]
fn pad_zero_pad() {
    // create image data
    let data_2d = linear_gradient_2d(1, 100.0, (20, 20));
    let data_3d = linear_gradient_3d(1, 100.0, (20, 20, 20));

    // pad test images isometrically with zeros
    let pad_2d = pad::zero_pad(&data_2d, &PAD_CONFIG_2D, None).unwrap();
    let pad_3d = pad::zero_pad(&data_3d, &PAD_CONFIG_3D, None).unwrap();

    // assert padded shape and contents are copied
    assert_eq!(pad_2d.shape(), &[30, 30]);
    assert_eq!(pad_3d.shape(), &[30, 30, 30]);
    assert_eq!(pad_2d[[2, 2]], 0.0);
    assert_eq!(pad_3d[[2, 2, 2]], 0.0);
    assert_eq!(pad_2d[[24, 24]], 1800.0);
    assert_eq!(pad_3d[[24, 24, 24]], 1800.0);
}
