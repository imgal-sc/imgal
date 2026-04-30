use std::array;

use ndarray::{ArrayBase, AsArray, Ix1, ViewRepr};

use crate::traits::numeric::AsNumeric;

/// Compute the 2D orientation predicate of a triangle.
///
/// # Description
///
/// Computes the 2D orientation predicate of triangle `(o, a, b)` in
/// `(row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented line `(o, a)` the point `b` lies:
///
/// - *Positive*: Point `b` lies to the left of the directed line `o -> a`
///   (counterclockwise turn).
/// - *Negative*: Point `b` lies to the right of the directed line `o -> a`
///   (clockwise turn).
/// - *Zero*: Points `o`, `a`, and `b` are collinear.
///
/// # Arguments
///
/// * `o`: The origin vertex of the directed line.
/// * `a`: The endpoint vertex of the directed line.
/// * `b`: The reference point relative to line `(o, a)`.
///
/// # Returns
///
/// * `f64`: The orientation of the triangle.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
/// <https://doi.org/10.1007/PL00009321>
pub fn orient_pred_2d<'a, T, A>(o: A, a: A, b: A) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let o: ArrayBase<ViewRepr<&'a T>, Ix1> = o.into();
    let a: ArrayBase<ViewRepr<&'a T>, Ix1> = a.into();
    let b: ArrayBase<ViewRepr<&'a T>, Ix1> = b.into();
    let [oy, ox] = array::from_fn(|i| o[i].to_f64());
    let [ay, ax] = array::from_fn(|i| a[i].to_f64());
    let [by, bx] = array::from_fn(|i| b[i].to_f64());
    (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
}

/// Compute the 3D orientation predicate of a tetrahedron.
///
/// # Description
///
/// Computes the 3D orientation prpedicate of tetrahedrion `(a, b, c, d)` in
/// `(pln, row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented plane `(a, b, c)` the point `d` lies:
///
/// - *Positive*: Point `d` lies above the plane where `(a, b, c)` appears in
///   counterclockwise order.
/// - *Negative*: Point `d` lies below the plane where `(a, b, c)` appears in
///   clockwise order.
/// - *Zero*: Point `d` is coplanar with plane `(a, b, c)`.
///
/// # Arguments
///
/// * `a`: Vertex `a` of the oriented plane.
/// * `b`: Vertex `b` of the oriented plane.
/// * `c`: Vertex `c` of the oriented plane.
/// * `d`: The reference point relative to plane `(a, b, c)`.
///
/// # Returns
///
/// * `f64`: The orientation of the tetrahedron.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
/// <https://doi.org/10.1007/PL00009321>
#[inline]
pub fn orient_pred_3d<'a, T, A>(a: A, b: A, c: A, d: A) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let a: ArrayBase<ViewRepr<&'a T>, Ix1> = a.into();
    let b: ArrayBase<ViewRepr<&'a T>, Ix1> = b.into();
    let c: ArrayBase<ViewRepr<&'a T>, Ix1> = c.into();
    let d: ArrayBase<ViewRepr<&'a T>, Ix1> = d.into();
    let [az, ay, ax] = array::from_fn(|i| a[i].to_f64());
    let [bz, by, bx] = array::from_fn(|i| b[i].to_f64());
    let [cz, cy, cx] = array::from_fn(|i| c[i].to_f64());
    let [dz, dy, dx] = array::from_fn(|i| d[i].to_f64());
    let [adx, ady, adz] = [ax - dx, ay - dy, az - dz];
    let [bdx, bdy, bdz] = [bx - dx, by - dy, bz - dz];
    let [cdx, cdy, cdz] = [cx - dx, cy - dy, cz - dz];
    adx * (bdy * cdz - bdz * cdy) - ady * (bdx * cdz - bdz * cdx) + adz * (bdx * cdy - bdy * cdx)
}

/// Compute the signed volume of a tetrahedron.
///
/// # Description
///
/// Computes the signed volume of tetrahedron `(a, b, c, d)` in
/// `(pln, row, col)` dimension order. The sign of the returned value indicates
/// on which side of the oriented plane `(a, b, c)` the point `d` lies:
///
/// - *Positive*: Point `d` lies above the plane where `(a, b, c)` appears in
///   counterclockwise order.
/// - *Negative*: Point `d` lies below the plane where `(a, b, c)` appears in
///   clockwise order.
/// - *Zero*: Point `d` is coplanar with plane `(a, b, c)`.
///
/// # Arguments
///
/// * `a`: Vertex `a` of the oriented plane.
/// * `b`: Vertex `b` of the oriented plane.
/// * `c`: Vertex `c` of the oriented plane.
/// * `d`: The reference point relative to plane `(a, b, c)`.
///
/// # Returns
///
/// * `f64`: The signed volume of the tetrahedron.
///
/// # Reference
///
/// <https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c>
/// <https://doi.org/10.1007/PL00009321>
#[inline]
pub fn tetrahedron_volume<'a, T, A>(a: A, b: A, c: A, d: A) -> f64
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    orient_pred_3d(a, b, c, d) / 6.0
}
