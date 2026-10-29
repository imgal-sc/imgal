use cubecl::prelude::*;
use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr, Zip};
use rayon::prelude::*;

use crate::gpu::*;
use crate::prelude::*;
use crate::simd_hint::fast_fold;

/// Find the maximum value in an n-dimensional image.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional image to determine the
/// maximum value.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok(T)`: The maximum value in the input n-dimensional image.
/// * `Err(ImgalError)`: If `data.is_empty() == true`.
#[inline(always)]
pub fn max<'a, T, A, D>(data: A, threads: Option<usize>) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let max_op = |acc, v| if v > acc { v } else { acc };
    let av = *data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    Ok(par!(threads,
        seq_exp: fast_fold(data, || av, max_op),
        par_exp: Zip::from(data.rows()).into_par_iter()
            .fold(|| av, |acc, (r,)| {
                let m = fast_fold(r, || av, max_op);
                max_op(acc, m)
            }).reduce(|| av, max_op)))
}

/// Find the minimum value in an n-dimensional image.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional image to determine the
/// minimum value.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok(T)`: The minimum value in the input n-dimensional image.
/// * `Err(ImgalError)`: If `data.is_empty() == true`.
#[inline(always)]
pub fn min<'a, T, A, D>(data: A, threads: Option<usize>) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let min_op = |acc, v| if v < acc { v } else { acc };
    let av = *data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    Ok(par!(threads,
        seq_exp: fast_fold(data, || av, min_op),
        par_exp: Zip::from(data.rows()).into_par_iter()
            .fold(|| av, |acc, (r,)| {
                let m = fast_fold(r, || av, min_op);
                min_op(acc, m)
            }).reduce(|| av, min_op)))
}

/// Find the minimum and maximum values in an n-dimensional image.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional image to determine the
/// minimum and maximum values.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image.
/// * `threads`: The requested number of threads to use for parallel execution.
///   If `None` or `Some(1)` sequential execution is used. If `Some(0)`, then
///   the maximum available parallelism is used. Thread counts are clamped to
///   the systems maximum.
///
/// # Returns
///
/// * `Ok((T, T))`: A tuple containing the minimum and maximum values (*i.e.*
///   (min, max)) in the given n-dimensional image.
/// * `Err(ImgalError)`: If `data.is_empty() == true`.
#[inline(always)]
pub fn min_max<'a, T, A, D>(data: A, threads: Option<usize>) -> Result<(T, T), ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let max_op = |acc, v| if v > acc { v } else { acc };
    let min_op = |acc, v| if v < acc { v } else { acc };
    let min_max_op = |acc: (T, T), v: (T, T)| {
        (
            if v.0 < acc.0 { v.0 } else { acc.0 },
            if v.1 > acc.1 { v.1 } else { acc.1 },
        )
    };
    let av = *data
        .first()
        .ok_or(ImgalError::InvalidParameterEmptyArray { param_name: "data" })?;
    Ok(par!(threads,
        seq_exp: Zip::from(data.rows()).fold((av, av), |acc, r| {
            let res = (fast_fold(r, || av, min_op), fast_fold(r, || av, max_op));
            min_max_op(acc, res)
            }),
        par_exp: Zip::from(data.rows()).into_par_iter()
            .fold(|| (av, av), |acc, (r,)| {
                let res = (fast_fold(r, || av, min_op), fast_fold(r, || av, max_op));
                min_max_op(acc, res)
            }).reduce(|| (av, av), min_max_op)))
}

/// Find the minimum value of an *n*-dimensional image on the GPU.
///
/// # Description
///
/// Iterates through all elements of an *n*-dimensional to determine the
/// maximum value on the GPU
///
/// # Arguments
///
/// * `data`: The input *n*-dimensional image as a tensor.
///
/// # Returns
///
/// * `N`: The minimum value.
#[inline(always)]
pub fn min_gpu<N>(data: &GpuTensor<N>) -> N
where
    N: Numeric + CubeElement,
{
    let client = GPU_CLIENT.get().expect(GPU_CLIENT_FAIL_MSG);
    let (cube_count, cube_dim, counts) = calculate_cube_config(data.size, NumCubeDims::One, None);
    let partials = GpuTensor::<N>::empty(&[counts.0]);
    let res = GpuTensor::<N>::empty_one();
    unsafe {
        min_gpu_kernel::launch_unchecked::<N>(
            client,
            cube_count,
            cube_dim,
            data.as_arg(),
            partials.as_arg(),
            data.size,
            cube_dim.x as usize,
        );
        min_gpu_kernel::launch_unchecked::<N>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_single(),
            partials.as_arg(),
            res.as_arg(),
            partials.size,
            1,
        );
    }
    res.read()[0]
}

/// Compute the minimum value of a tensor.
///
/// # Description
///
/// Computes the minimum value of an input tensor.
///
/// # Arguments
///
/// * `input`: The input tensor.
/// * `output`: The output tensor.
/// * `tensor_size`: The size of the input tensor (*i.e.* number of elements).
/// * `cube_size`: The size of the dispatched cubes (*i.e.* cube dim).
#[cube(launch_unchecked)]
pub(crate) fn min_gpu_kernel<N: Numeric>(
    input: &Tensor<N>,
    output: &mut Tensor<N>,
    tensor_size: usize,
    #[comptime] cube_size: usize,
) {
    let stride = CUBE_COUNT * (CUBE_DIM as usize);
    let mut i = ABSOLUTE_POS;
    let mut acc = input[i];
    while i < tensor_size {
        let v = input[i];
        if v < acc {
            acc = v
        };
        i += stride;
    }
    let mut buf: Shared<[N]> = Shared::new_slice(cube_size);
    buf[UNIT_POS as usize] = acc;
    sync_cube();
    if UNIT_POS == 0 {
        let mut cube_res = buf[0];
        for j in 0..cube_size {
            if buf[j] < cube_res {
                cube_res = buf[j]
            };
        }
        output[CUBE_POS] = cube_res;
    }
}
