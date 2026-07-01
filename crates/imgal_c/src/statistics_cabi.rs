use std::slice;

use ndarray::{ArrayViewD, IxDyn};

use imgal::prelude::*;
use imgal::statistics;

#[unsafe(no_mangle)]
pub extern "C" fn max(ptr: *const f64, len: usize, threads: usize) -> f64 {
    // validate the pointer and array length
    if ptr.is_null() || len == 0 {
        return 0.0;
    }

    // create slice from pointer and len, compute max
    let data = unsafe { slice::from_raw_parts(ptr, len) };
    let shape = IxDyn(&[data.len()]);
    let arr = ArrayViewD::from_shape(shape, data);

    // TODO: Yikes. If we have an empty input array and issue an ImgalError what happens
    // here?
    statistics::max(arr.unwrap(), Some(threads)).unwrap()
}

#[unsafe(no_mangle)]
pub extern "C" fn sum_u8(data_ptr: *const u8, data_len: usize, threads: usize) -> u8 {
    sum_generic(data_ptr, data_len, threads)
}

#[unsafe(no_mangle)]
pub extern "C" fn sum_u16(data_ptr: *const u16, data_len: usize, threads: usize) -> u16 {
    sum_generic(data_ptr, data_len, threads)
}

#[unsafe(no_mangle)]
pub extern "C" fn sum_u64(data_ptr: *const u64, data_len: usize, threads: usize) -> u64 {
    sum_generic(data_ptr, data_len, threads)
}

#[unsafe(no_mangle)]
pub extern "C" fn sum_i32(data_ptr: *const i32, data_len: usize, threads: usize) -> i32 {
    sum_generic(data_ptr, data_len, threads)
}

#[unsafe(no_mangle)]
pub extern "C" fn sum_i64(data_ptr: *const i64, data_len: usize, threads: usize) -> i64 {
    sum_generic(data_ptr, data_len, threads)
}

#[unsafe(no_mangle)]
pub extern "C" fn sum_f32(data_ptr: *const f32, data_len: usize, threads: usize) -> f32 {
    sum_generic(data_ptr, data_len, threads)
}

#[unsafe(no_mangle)]
pub extern "C" fn sum_f64(data_ptr: *const f64, data_len: usize, threads: usize) -> f64 {
    sum_generic(data_ptr, data_len, threads)
}

/// TODO
fn sum_generic<T>(data_ptr: *const T, data_len: usize, threads: usize) -> T
where
    T: AsNumeric,
{
    if data_ptr.is_null() || data_len == 0 {
        return T::from_i32(-1);
    }
    let s = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    statistics::sum(&s, Some(threads))
}
