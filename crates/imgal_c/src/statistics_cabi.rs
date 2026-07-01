use std::slice;

use ndarray::{ArrayViewD, IxDyn};

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
pub extern "C" fn sum(data_ptr: *const f64, data_len: usize, threads: usize) -> f64 {
    if data_ptr.is_null() || data_len == 0 {
        return 0.0;
    }
    let s = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    statistics::sum(&s, Some(threads))
}
