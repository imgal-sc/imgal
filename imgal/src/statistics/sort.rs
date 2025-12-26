use std::cmp::Ordering;

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Sort 1-dimensional arrays of values and their associated weights.
///
/// # Description
///
/// Performs a bottom up merge sort on the input 1-dimensional data array along
/// with it's associated weights. Both the `data` and `weights` arrays are
/// _mutated_ during the sorting. The output of this function is a weighted
/// inversion count.
///
/// # Arguments
///
/// * `data`: A 1-dimensional array/slice of numbers of the same length as
///   `weights`.
/// * `weights`: A 1-dimensional array/slice of weights of the same length as
///   `data`.
///
/// # Returns
///
/// * `OK(f64)`: The number of swaps needed to sort the input array.
/// * `Err(ImgalError)`: If the `data.len() != weights.len()`.
///
/// # Reference
///
/// <https://doi.org/10.1109/TIP.2019.2909194>
pub fn weighted_merge_sort_mut<T>(data: &mut [T], weights: &mut [f64]) -> Result<f64, ImgalError>
where
    T: AsNumeric,
{
    // validate the input arrays are the same length
    let dl = data.len();
    let wl = weights.len();
    if dl != wl {
        return Err(ImgalError::MismatchedArrayLengths {
            a_arr_name: "data",
            a_arr_len: dl,
            b_arr_name: "weights",
            b_arr_len: wl,
        });
    };

    // create sort parameters and working buffers, start weighted bottom-up
    // merge sort
    let mut swap = 0.0;
    let mut swap_temp: f64;
    let mut step: usize = 1;
    let mut left: usize;
    let mut right: usize;
    let mut end: usize;
    let mut k: usize;
    let mut data_buf = vec![T::default(); dl];
    let mut weights_buf = vec![0.0; dl];
    let mut cum_weights_buf = vec![0.0; dl];
    while step < dl {
        left = 0;
        k = 0;
        let mut cw_acc = weights[0];
        cum_weights_buf[0] = weights[0];
        cum_weights_buf
            .iter_mut()
            .zip(weights.iter())
            .skip(1)
            .for_each(|(cw, w)| {
                *cw = cw_acc + w;
                cw_acc = *cw;
            });

        loop {
            right = left + step;
            end = right + step;
            if end > dl {
                if right > dl {
                    break;
                }
                end = dl;
            }
            let mut l = left;
            let mut r = right;
            while l < right && r < end {
                match data[l].partial_cmp(&data[r]) {
                    Some(Ordering::Greater) => {
                        if l == 0 {
                            swap_temp = weights[r] * cum_weights_buf[right - 1];
                        } else {
                            swap_temp =
                                weights[r] * (cum_weights_buf[right - 1] - cum_weights_buf[l - 1]);
                        }
                        swap += swap_temp;
                        data_buf[k] = data[r];
                        weights_buf[k] = weights[r];
                        k += 1;
                        r += 1;
                    }
                    _ => {
                        data_buf[k] = data[l];
                        weights_buf[k] = weights[l];
                        k += 1;
                        l += 1;
                    }
                }
            }
            if l < right {
                while l < right {
                    data_buf[k] = data[l];
                    weights_buf[k] = weights[l];
                    k += 1;
                    l += 1;
                }
            } else {
                while r < end {
                    data_buf[k] = data[r];
                    weights_buf[k] = weights[r];
                    k += 1;
                    r += 1;
                }
            }
            left = end;
        }

        // copy any unmerged tail, if array size is not a power of 2
        if k < dl {
            while k < dl {
                data_buf[k] = data[k];
                weights_buf[k] = weights[k];
                k += 1;
            }
        }

        // prepare for the next step, copy merged results back source
        data.clone_from_slice(&data_buf);
        weights.clone_from_slice(&weights_buf);

        // double the run size, continue
        step *= 2;
    }

    Ok(swap)
}
