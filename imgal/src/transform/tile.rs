use ndarray::{ArrayBase, ArrayD, ArrayView, AsArray, Axis, Dimension, IxDyn, Slice, ViewRepr};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Tile an n-dimensional array using division tiling.
///
/// # Description
///
/// Divides an n-dimensional array into a regular grid of tiles and returns them
/// as a vector of views. The array is divided along each axis into `div`
/// equally sized segments per axis. This produces a total of `divⁿ` tiles for
/// a given array, where `n` is the total number of dimensions. This function is
/// *naive* in that it does not produce tiles intended for image fusing. Instead
/// the tiles are simple regular slices of the input data.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array to be tiled.
/// * `div`: The base number of divisions ber paxis. This value must be `>0`.
///
/// # Returns
///
/// * `Ok(Vec<ArrayView<'a, T, D>>)`: A vector containing views of all tiles in
///   row-major order. The length of the vector will be `divⁿ`, the number of
///   tiles.
/// * `Err(ImgalError)`: If `div == 0`. If an axis length is not a multiple of
///   `div`.
pub fn div_tile<'a, T, A, D>(data: A, div: usize) -> Result<Vec<ArrayView<'a, T, D>>, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    if div == 0 {
        return Err(ImgalError::InvalidParameterValueEqual {
            param_name: "div",
            value: 0,
        });
    }
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let view_shape = view.shape().to_vec();
    view_shape
        .iter()
        .enumerate()
        .filter(|&(_, &v)| !v.is_multiple_of(div))
        .try_for_each(|(i, &v)| {
            Err(ImgalError::InvalidAxisValueNotAMultipleOf {
                arr_name: "shape",
                axis_idx: i,
                multiple: v,
            })
        })?;
    let n_dims = view.shape().len();
    let tile_positions: Vec<Vec<(isize, isize)>> = view_shape
        .iter()
        .map(|&v| get_div_start_stop_positions(div, v))
        .collect();
    let n_tiles: usize = tile_positions.iter().map(|v| v.len()).product();
    let mut tile_stack: Vec<ArrayView<T, D>> = Vec::with_capacity(n_tiles);
    (0..n_tiles).for_each(|t| {
        let mut tile_view = view.clone();
        let mut remaining = t;
        (0..n_dims).for_each(|a| {
            let stride: usize = tile_positions.iter().skip(a + 1).map(|v| v.len()).product();
            let tile_pos = remaining / stride;
            remaining %= stride;
            let ax_slice = Slice {
                start: tile_positions[a][tile_pos].0,
                end: Some(tile_positions[a][tile_pos].1),
                step: 1,
            };
            tile_view.slice_axis_inplace(Axis(a), ax_slice);
        });
        tile_stack.push(tile_view);
    });

    Ok(tile_stack)
}

/// Untile a tile stack into an n-dimensional array.
///
/// # Description
///
/// Reconstructs (*.i.e.* untiles) an n-dimensional array by assembling a stack
/// of equally sized n-dimensional tiles into a single output array of the given
/// `shape`. The input `tile_stack` is assumed to contain tiles resulting from
/// the `div_tile` function or a similar tiling scheme where tiles are stored in
/// row-major order. This function is *naive* in that it does not offer any
/// border fusing strategies.
///
/// # Arguments
///
/// * `tile_stack`: A vector containing views (*i.e.* tiles) to be reassembled
///   into a single array.
/// * `div`: The base number of divisions ber paxis. This value must be `>0`.
/// * `shape`: The shape of the output array. Its dimensionality must match the
///   dimensionality of the tiles. Each axis length must be a multiple of `div`.
///
/// # Returns
///
/// * `Ok(ArrayD<T>)`: An n-dimensional array with the given `shape` containing
///   all tiles in their corresponding positions.
/// * `Err(ImgalError)`: If `tile_stack.is_empty() == true`. If `div == 0`. If
///   an axis length of `shape` is not a multiple of `div`. If `shape.len()` is
///   not equal to the tile shape length. If expected tile shapes do not match
///   given tile shapes. If the number of tiles given does not match the number
///   of tiles expected.
pub fn div_untile<'a, T, D>(
    tile_stack: Vec<ArrayView<'a, T, D>>,
    div: usize,
    shape: &[usize],
) -> Result<ArrayD<T>, ImgalError>
where
    D: Dimension,
    T: 'a + AsNumeric,
{
    if tile_stack.is_empty() {
        return Err(ImgalError::InvalidParameterEmptyArray {
            param_name: "tile_stack",
        });
    }
    if div == 0 {
        return Err(ImgalError::InvalidParameterValueEqual {
            param_name: "div",
            value: 0,
        });
    }
    let n_dims = tile_stack[0].shape().len();
    if shape.len() != n_dims {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "tile shape",
            a_arr_len: n_dims,
            b_arr_name: "shape",
            b_arr_len: shape.len(),
        });
    }
    shape
        .iter()
        .enumerate()
        .filter(|&(_, &v)| !v.is_multiple_of(div))
        .try_for_each(|(i, _)| {
            Err(ImgalError::InvalidAxisValueNotAMultipleOf {
                arr_name: "shape",
                axis_idx: i,
                multiple: div,
            })
        })?;
    let tile_positions: Vec<Vec<(isize, isize)>> = shape
        .iter()
        .map(|&v| get_div_start_stop_positions(div, v))
        .collect();
    let n_tiles: usize = tile_positions.iter().map(|v| v.len()).product();
    if n_tiles != tile_stack.len() {
        return Err(ImgalError::InvalidArrayLength {
            arr_name: "tile_stack",
            expected: n_tiles,
            got: tile_stack.len(),
        });
    }
    let tile_shape: Vec<usize> = shape.iter().map(|&v| v / div).collect();
    if tile_shape != tile_stack[0].shape() {
        return Err(ImgalError::MismatchedArrayShapes {
            a_arr_name: "expected tile",
            a_shape: tile_shape,
            b_arr_name: "input tile",
            b_shape: tile_stack[0].shape().to_vec(),
        });
    }
    let mut untile_arr: ArrayD<T> = ArrayD::from_elem(IxDyn(&shape), T::default());
    (0..n_tiles).for_each(|t| {
        let tile_view = tile_stack[t].view();
        let mut untile_view = untile_arr.view_mut();
        let mut remaining = t;
        (0..n_dims).for_each(|a| {
            let stride: usize = tile_positions.iter().skip(a + 1).map(|v| v.len()).product();
            let tile_pos = remaining / stride;
            remaining %= stride;
            let ax_slice = Slice {
                start: tile_positions[a][tile_pos].0,
                end: Some(tile_positions[a][tile_pos].1),
                step: 1,
            };
            untile_view.slice_axis_inplace(Axis(a), ax_slice);
        });
        untile_view.assign(&tile_view);
    });

    Ok(untile_arr)
}

/// Compute evenly spaced start and stop positions
///
/// # Arguments
///
/// * `div`: The base number of divisions ber paxis. This value must be `>0`.
/// * `axis_len`: The length of the axis to compute start and stop positions.
///   This function assumes that `axis_len.is_multiple_of(div) == true`.
///
/// # Returns
///
/// * `Vec<(isize, isize)>`: A tuple of start and stop positions,
///   `(start, stop)` along an axis.
fn get_div_start_stop_positions(div: usize, axis_len: usize) -> Vec<(isize, isize)> {
    let mut start_stop_arr: Vec<(isize, isize)> = Vec::with_capacity(div);
    let inc = (axis_len / div) as isize;
    (0..div).fold(0_isize, |acc, _| {
        let start = acc;
        let stop = acc + inc;
        start_stop_arr.push((start, stop));
        stop
    });

    start_stop_arr
}
