use pyo3::PyErr;
use pyo3::exceptions::{PyException, PyIndexError, PyValueError};

use imgal::error::ImgalError;

/// Map ImgalError types to Python exceptions.
pub fn map_imgal_error(err: ImgalError) -> PyErr {
    match err {
        ImgalError::InvalidAxis { axis_idx, dim_len } => PyIndexError::new_err(format!(
            "Axis {} is out of bounds for dimension length {}.",
            axis_idx, dim_len
        )),
        ImgalError::InvalidAxisValueGreaterEqual {
            arr_name,
            axis_idx,
            value,
        } => PyIndexError::new_err(format!(
            "Invalid axis value, axis {} of \"{}\" can not be greater than or equal to {}.",
            axis_idx, arr_name, value
        )),
        ImgalError::InvalidGeneric { msg } => PyException::new_err(format!("{}", msg)),
        ImgalError::InvalidParameterEmptyArray { param_name } => PyException::new_err(format!(
            "Invalid array parameter, the array \"{}\" can not be empty.",
            param_name
        )),
        ImgalError::InvalidParameterValueEqual { param_name, value } => {
            PyValueError::new_err(format!(
                "Invalid parameter value, the parameter {} can not equal {}.",
                param_name, value
            ))
        }
        ImgalError::InvalidParameterValueGreater { param_name, value } => {
            PyValueError::new_err(format!(
                "Invalid parameter value, the parameter {} can not be greater than {}.",
                param_name, value
            ))
        }
        ImgalError::InvalidParameterValueLess { param_name, value } => {
            PyValueError::new_err(format!(
                "Invalid parameter value, the parameter {} can not be less than {}.",
                param_name, value
            ))
        }
        ImgalError::InvalidParameterValueOutsideRange {
            param_name,
            value,
            min,
            max,
        } => PyValueError::new_err(format!(
            "Invalid parameter value, the parameter {} must be a value between {} and {} but got {}.",
            param_name, min, max, value
        )),
        ImgalError::InvalidSum { expected, got } => PyValueError::new_err(format!(
            "Invalid sum, expected {} but got {}.",
            expected, got
        )),
        ImgalError::MismatchedArrayLengths {
            a_arr_name,
            a_arr_len,
            b_arr_name,
            b_arr_len,
        } => PyValueError::new_err(format!(
            "Mismatched array lengths, \"{}\" of length {} and \"{}\" of length {} do not match.",
            a_arr_name, a_arr_len, b_arr_name, b_arr_len
        )),
        ImgalError::MismatchedArrayShapes { shape_a, shape_b } => PyValueError::new_err(format!(
            "Mismatched array shapes, {:?} and {:?}, do not match.",
            shape_a, shape_b
        )),
    }
}
