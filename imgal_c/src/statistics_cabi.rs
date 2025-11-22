use std::slice;

use ndarray::{ArrayViewD, IxDyn};

use imgal::statistics;

#[unsafe(no_mangle)]
pub extern "C" fn max(ptr: *const f64, len: usize) -> f64 {
    // validate the pointer and array length
    if ptr.is_null() || len == 0 {
        return 0.0;
    }

    // create slice from pointer and len, compute max
    let data = unsafe { slice::from_raw_parts(ptr, len) };
    let shape = IxDyn(&[data.len()]);
    let arr = ArrayViewD::from_shape(shape, data);

    statistics::max(arr.unwrap())
}

#[unsafe(no_mangle)]
pub extern "C" fn sum(ptr: *const f64, len: usize) -> f64 {
    // saftey check: validate the pointer and array length
    if ptr.is_null() || len == 0 {
        return 0.0;
    }
    // create a slice and compute sum
    let s = unsafe { slice::from_raw_parts(ptr, len) };
    statistics::sum(&s)
}
