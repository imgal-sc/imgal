use imgal::kernel::neighborhood;

// kernel parameters
const RADIUS: usize = 5;
const FALLOFF_RADIUS: f64 = 7.0;

#[test]
fn neighborhood_circle_kernel() {
    // create a circle neighborhood kernel
    let k = neighborhood::circle_kernel(RADIUS).unwrap();

    // check the kernel shape, kernel center, a point inside the shape, and
    // a point outside (background)
    assert_eq!(k.shape(), [11, 11]);
    assert_eq!(k[[RADIUS, RADIUS]], true);
    assert_eq!(k[[8, 1]], true);
    assert_eq!(k[[2, 0]], false);
}

#[test]
fn neighborhood_sphere_kernel() {
    // create a sphere neighborhood kernel
    let k = neighborhood::sphere_kernel(RADIUS).unwrap();

    // check the kernel shape, kernel center, a point inside the shape, and
    // a point outside (background)
    assert_eq!(k.shape(), [11, 11, 11]);
    assert_eq!(k[[RADIUS, RADIUS, RADIUS]], true);
    assert_eq!(k[[2, 5, 1]], true);
    assert_eq!(k[[8, 9, 10]], false);
}

#[test]
fn neighborhood_weighted_circle_kernel() {
    // create a weighted circle neighborhood kernel
    let k = neighborhood::weighted_circle_kernel(RADIUS, FALLOFF_RADIUS, None).unwrap();

    // check the kernel shape, kernel center, a point inside the shape, and
    // a point outside (background)
    assert_eq!(k.shape(), [11, 11]);
    assert_eq!(k[[RADIUS, RADIUS]], 1.0);
    assert_eq!(k[[8, 1]], 0.2857142857142857);
    assert_eq!(k[[2, 0]], 0.0);
}

#[test]
fn neighborhood_weighted_sphere_kernel() {
    // create a weighted sphere neighborhood kernel
    let k = neighborhood::weighted_sphere_kernel(RADIUS, FALLOFF_RADIUS, None).unwrap();

    // check the kernel shape, kernel center, a point inside the shape, and
    // a point outside (background)
    assert_eq!(k.shape(), [11, 11, 11]);
    assert_eq!(k[[RADIUS, RADIUS, RADIUS]], 1.0);
    assert_eq!(k[[2, 5, 1]], 0.2857142857142857);
    assert_eq!(k[[8, 9, 10]], 0.0);
}
