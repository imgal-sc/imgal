use imgal::error::ImgalError;
use imgal::kernel::neighborhood::{
    circle_kernel, sphere_kernel, weighted_circle_kernel, weighted_sphere_kernel,
};

const TOLERANCE: f64 = 1e-10;
const RADIUS: usize = 5;
const FALLOFF_RADIUS: f64 = 7.0;

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < TOLERANCE
}

/// Tests that `circle_kernel` returns the expected kernel by checking the
/// shape and points inside/outside the circle.
#[test]
fn neighborhood_circle_kernel_expected_results() -> Result<(), ImgalError> {
    let k = circle_kernel(RADIUS)?;
    assert_eq!(k.shape(), [11, 11]);
    assert_eq!(k[[RADIUS, RADIUS]], true);
    assert_eq!(k[[8, 1]], true);
    assert_eq!(k[[2, 0]], false);
    Ok(())
}

/// Tests that `sphere_kernel` returns the expected kernel by checking the
/// shape and points inside/outside the sphere.
#[test]
fn neighborhood_sphere_kernel_expected_results() -> Result<(), ImgalError> {
    let k = sphere_kernel(RADIUS)?;
    assert_eq!(k.shape(), [11, 11, 11]);
    assert_eq!(k[[RADIUS, RADIUS, RADIUS]], true);
    assert_eq!(k[[2, 5, 1]], true);
    assert_eq!(k[[8, 9, 10]], false);
    Ok(())
}

/// Tests that `weighted_circle_kernel` returns the expected weighted kernel by
/// checking the shape and values inside/outside the circle.
#[test]
fn neighborhood_weighted_circle_kernel_expected_results() -> Result<(), ImgalError> {
    let k = weighted_circle_kernel(RADIUS, FALLOFF_RADIUS, None)?;
    assert_eq!(k.shape(), [11, 11]);
    assert_eq!(k[[RADIUS, RADIUS]], 1.0);
    assert_eq!(k[[2, 0]], 0.0);
    assert!(approx_equal(k[[8, 1]], 0.2857142857));
    Ok(())
}

/// Tests that `weighted_sphere_kernel` returns the expected weighted kernel by
/// checking the shape and values inside/outside the sphere.
#[test]
fn neighborhood_weighted_sphere_kernel_expected_results() -> Result<(), ImgalError> {
    let k = weighted_sphere_kernel(RADIUS, FALLOFF_RADIUS, None)?;
    assert_eq!(k.shape(), [11, 11, 11]);
    assert_eq!(k[[RADIUS, RADIUS, RADIUS]], 1.0);
    assert_eq!(k[[8, 9, 10]], 0.0);
    assert!(approx_equal(k[[2, 5, 1]], 0.2857142857));
    Ok(())
}
