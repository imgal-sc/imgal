use ndarray::{Array1, Array2, ArrayBase, ArrayView1, AsArray, Axis, Ix1, Ix2, ViewRepr, stack};
use rayon::prelude::*;

use crate::prelude::*;
use crate::spatial::convex_hull::graham_scan;

/// Convert the vertices of an edge into halfplane representation.
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// todo
///
/// # Returns
///
/// * `Ok(Array1<f64>)`: The vector `[Ny, Nx, d]` describing the halfplane.
///
/// # Reference
///
/// todo
#[inline(always)]
pub fn edge_to_halfplane<'a, T, A>(a: A, b: A) -> Result<Array1<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let a: ArrayBase<ViewRepr<&'a T>, Ix1> = a.into();
    let b: ArrayBase<ViewRepr<&'a T>, Ix1> = b.into();
    if a.len() != 2 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "a",
            expected: 2,
            got: a.len(),
        });
    }
    if b.len() != 2 {
        return Err(ImgalError::InvalidArrayLengthExpected {
            arr_name: "b",
            expected: 2,
            got: b.len(),
        });
    }
    let a_pnt: [f64; 2] = [a[0].to_f64(), a[1].to_f64()];
    let b_pnt: [f64; 2] = [b[0].to_f64(), b[1].to_f64()];
    let py = b_pnt[0] - a_pnt[0];
    let px = b_pnt[1] - a_pnt[1];
    // here we rotate the normal 90 deg clockwise outward
    let ny = -px;
    let nx = py;
    let d = -(ny * a_pnt[0] + nx * a_pnt[1]);
    Ok(Array1::from_vec(vec![ny, nx, d]))
}

/// TODO
#[inline(always)]
pub fn halfplane_intersection<'a, T, A, B>(
    halfplanes: A,
    interior_point: B,
    threads: Option<usize>,
) -> Result<Array2<f64>, ImgalError>
where
    A: AsArray<'a, f64, Ix2>,
    B: AsArray<'a, T, Ix1>,
    T: 'a + AsNumeric,
{
    let halfplanes: ArrayBase<ViewRepr<&'a f64>, Ix2> = halfplanes.into();
    let int_pnt: ArrayBase<ViewRepr<&'a T>, Ix1> = interior_point.into();
    if halfplanes.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "halfplanes",
        });
    }
    if halfplanes.dim().1 != 3 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "halfplanes",
            axis_idx: 1,
            expected: 3,
            got: halfplanes.dim().1,
        });
    }
    let n_hp = halfplanes.dim().0;
    let [qy, qx] = [int_pnt[0].to_f64(), int_pnt[1].to_f64()];
    // we start by converting each halfplane normal vector (primal space) into
    // dual points (dual space)
    let mut dual_points = Array2::<f64>::zeros((n_hp, 2));
    (0..n_hp).try_for_each(|i| {
        let hp = halfplanes.row(i);
        let mut dp = dual_points.row_mut(i);
        let [ny, nx, d] = [hp[0], hp[1], hp[2]];
        let cur_d = ny * qy + nx * qx + d;
        if cur_d.abs() < 1e-12 {
            return Err(ImgalError::InvalidGeneric {
                msg: "The interior point lies on a halfplane boundary.",
            });
        }
        dp[0] = ny  / -cur_d;
        dp[1] = nx  / -cur_d;
        Ok(())
    })?;
    // constructing a convex hull of the dual points finds the intersection
    // vertices in primal space, the rest of the work is converting dual space
    // back into primal space
    let dual_verts = graham_scan(&dual_points, threads)?;
    let n_dv = dual_verts.dim().0;
    let primal_verts: Vec<f64> = (0..n_dv).fold(Vec::with_capacity(n_dv * 2), |mut acc, i|{
        let a = dual_verts.row(i);
        let b = dual_verts.row((i + 1) % n_dv);
        let [ay, ax] = [a[0], a[1]];
        let [by, bx] = [b[0], b[1]];
        let [dy, dx] = [by - ay, bx - ax];
        let ny = dx;
        let nx = -dy;
        let offset = ny * ay + nx * ax;
        // skip degenerate edages
        if offset.abs() < 1e-12 {
            return acc;
        }
        acc.push((ny / offset) + qy);
        acc.push((nx / offset) + qx);
        acc
    });
    let n_pv = primal_verts.len() / 2;
    if n_pv < 3 {
        return Err(ImgalError::InvalidArrayLengthMinimum { arr_name: "primal_verts", arr_len: n_pv, min_len: 3 });
    }
    let primal_verts = Array2::from_shape_vec((n_pv, 2), primal_verts).unwrap();
    graham_scan(primal_verts.view(), threads)
}

/// TODO
#[inline(always)]
pub fn hull_to_halfplane<'a, T, A>(
    vertices: A,
    threads: Option<usize>,
) -> Result<Array2<f64>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let vertices: ArrayBase<ViewRepr<&'a T>, Ix2> = vertices.into();
    if vertices.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "vertices",
        });
    }
    if vertices.dim().1 != 2 {
        return Err(ImgalError::InvalidAxisLengthExpected {
            arr_name: "vertices",
            axis_idx: 1,
            expected: 2,
            got: vertices.dim().1,
        });
    }
    let n = vertices.dim().0;
    let halfplane_calc = |mut acc: Vec<Array1<f64>>, i: usize| {
        let a = vertices.row(i);
        let b = vertices.row((i + 1) % n);
        // SAFE: this unwrap is safe because we validated the inputs already
        acc.push(edge_to_halfplane(a, b).unwrap());
        acc
    };
    let hp: Vec<Array1<f64>> = par!(threads,
    seq_exp: (0..n).fold(Vec::with_capacity(n), halfplane_calc),
    par_exp: (0..n).into_par_iter().fold(Vec::new, halfplane_calc)
        .reduce(Vec::new, |mut hp_out, hp_thread| {
            hp_out.extend(hp_thread);
            hp_out
        }));
    Ok(stack(
        Axis(0),
        &hp.iter()
            .map(|v| v.view())
            .collect::<Vec<ArrayView1<f64>>>(),
    )
    .unwrap())
}
