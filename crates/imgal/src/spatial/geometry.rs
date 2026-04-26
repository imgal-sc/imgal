use ndarray::ArrayView2;

use crate::traits::numeric::AsNumeric;

/// Computes the centroid of (*i.e.* the centroid) of the 3D coordinates.
pub fn centroid_3d<T>(points: &ArrayView2<T>, vertices: &[usize]) -> [f64; 3]
where
    T: AsNumeric,
{
    let n = vertices.len().max(1) as f64;
    let sum_verts = vertices.iter().fold([0.0_f64; 3], |acc, &i| {
        [
            acc[0] + points[[i, 0]].to_f64(),
            acc[1] + points[[i, 1]].to_f64(),
            acc[2] + points[[i, 2]].to_f64(),
        ]
    });
    [sum_verts[0] / n, sum_verts[1] / n, sum_verts[2] / n]
}

/// Compute the squared Euclidean distance between two 2D points.
///
/// # Arguments
///
/// * `point_a`: The first point as (row, col).
/// * `point_b`: The second point as (row, col).
///
/// # Returns
///
/// * `T`: The squared Euclidean distance.
pub fn dist_sq_2d<T>(point_a: &[T; 2], b: &[T; 2]) -> T
where
    T: AsNumeric,
{
    let dy = point_a[0] - b[0];
    let dx = point_a[1] - b[1];
    dx * dx + dy * dy
}

/// Compute the squared Euclidean distance between two 3D points.
///
/// # Arguments
///
/// * `a`: The first point as (pln, row, col).
/// * `b`: The second point as (pln, row, col).
///
/// # Returns
///
/// * `f64`: The squared Euclidean distance.
pub fn dist_sq_3d(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let abz = a[0] - b[0];
    let aby = a[1] - b[1];
    let abx = a[2] - b[2];
    (abz * abz) + (aby * aby) + (abx * abx)
}

/// Compute the 2D cross product of vectors defined by three points. This
/// function is also known as the orientiation predicate for the 2D case.
///
/// # Description
///
/// Calculates the cross product of vectors `(a - o)` and `(b - o)`. The result
/// indicates rotational direction:
///
/// - Positive => counterclockwise (left) turn
/// - Negative => clockwise (right) turn
/// - Zero => collinear points
///
/// # Arguments
///
/// * `o`: The origin point as (row, col).
/// * `a`: The first point as (row, col).
/// * `b`: The second point as (row, col).
///
/// # Returns
///
/// * `T`: The cross product.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
pub fn orient_pred_2d<T>(o: &[T; 2], a: &[T; 2], b: &[T; 2]) -> f64
where
    T: AsNumeric,
{
    (a[1].to_f64() - o[1].to_f64()) * (b[0].to_f64() - o[0].to_f64())
        - (a[0].to_f64() - o[0].to_f64()) * (b[1].to_f64() - o[1].to_f64())
}

/// Computes the 3D signed volume (*i.e.* orientation) of a tetrahedron.
///
/// # Description
///
/// The sign indicates the tetrahedron orientation:
/// - Positive => Point `d` is below the plane in CCW from outside the hull.
/// - Negative => Point `d` is above the plane in CCW from outside the hull.
/// - Zero => Point `d` lines on the plane (coplanar).
///
/// Note that this function assumes a "right-handed" system (X, Y, Z) which
/// means need to take the opposite sign when working in the "left-handed"
/// system (pln, row, col).
///
/// # Returns
///
/// * `f64`: The orientation of the tetrahedron.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
/// <https://doi.org/10.1007/PL00009321>
pub fn orient_pred_3d(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    let [adx, ady, adz] = [a[2] - d[2], a[1] - d[1], a[0] - d[0]];
    let [bdx, bdy, bdz] = [b[2] - d[2], b[1] - d[1], b[0] - d[0]];
    let [cdx, cdy, cdz] = [c[2] - d[2], c[1] - d[1], c[0] - d[0]];
    adx * (bdy * cdz - bdz * cdy) - ady * (bdx * cdz - bdz * cdx) + adz * (bdx * cdy - bdy * cdx)
}

/// Computes the squared area of the triangle defined by three 3D points `a`,
/// `b`, and `c` by taking the cross product of the edge vectors `ab = b - a`
/// `ac = c - a`.
///
/// # Returns
///
/// * `f64`: The squared area of the triangle (*i.e.* `4 * (area)^2`).
pub fn triangle_area_sq(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let [abx, aby, abz] = [b[2] - a[2], b[1] - a[1], b[0] - a[0]];
    let [acx, acy, acz] = [c[2] - a[2], c[1] - a[1], c[0] - a[0]];
    ((aby * acz - abz * acy) * (aby * acz - abz * acy))
        + ((abz * acx - abx * acz) * (abz * acx - abx * acz)
            + ((abx * acy - aby * acx) * (abx * acy - aby * acx)))
}
