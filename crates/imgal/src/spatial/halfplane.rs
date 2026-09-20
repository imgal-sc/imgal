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
pub fn halfplane_intersection() {
    todo!();
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
    let n = vertices.len();
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
