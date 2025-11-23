use imgal::simulation::gradient::{linear_gradient_2d, linear_gradient_3d};
use imgal::transform::pad;

#[test]
fn pad_isometric_pad_constant() {
    // create image data
    let data_2d = linear_gradient_2d(1, 100.0, (20, 20));
    let data_3d = linear_gradient_3d(1, 100.0, (20, 20, 20));

    // pad the images
    let pad_2d = pad::isometric_pad_constant(data_2d.into_dyn().view(), 900.0, 5);
    let pad_3d = pad::isometric_pad_constant(data_3d.into_dyn().view(), 900.0, 5);

    // assert padded shape and contents are copied
    assert_eq!(pad_2d.shape(), &[30, 30]);
    assert_eq!(pad_3d.shape(), &[30, 30, 30]);
    assert_eq!(pad_2d[[2, 2]], 900.0);
    assert_eq!(pad_3d[[2, 2, 2]], 900.0);
    assert_eq!(pad_2d[[24, 24]], 1800.0);
    assert_eq!(pad_3d[[24, 24, 24]], 1800.0);
}

#[test]
fn pad_isometric_pad_zero() {
    // create image data
    let data_2d = linear_gradient_2d(1, 100.0, (20, 20));
    let data_3d = linear_gradient_3d(1, 100.0, (20, 20, 20));

    // pad the images
    let pad_2d = pad::isometric_pad_zero(data_2d.into_dyn().view(), 5);
    let pad_3d = pad::isometric_pad_zero(data_3d.into_dyn().view(), 5);

    // assert padded shape and contents are copied
    assert_eq!(pad_2d.shape(), &[30, 30]);
    assert_eq!(pad_3d.shape(), &[30, 30, 30]);
    assert_eq!(pad_2d[[2, 2]], 0.0);
    assert_eq!(pad_3d[[2, 2, 2]], 0.0);
    assert_eq!(pad_2d[[24, 24]], 1800.0);
    assert_eq!(pad_3d[[24, 24, 24]], 1800.0);
}
