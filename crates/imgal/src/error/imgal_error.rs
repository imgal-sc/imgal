use std::error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ImgalError {
    InvalidAxis {
        axis_idx: usize,
        dim_len: usize,
    },
    InvalidArrayLengthExpected {
        arr_name: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidAxisValueGreaterEqual {
        arr_name: &'static str,
        axis_idx: usize,
        value: usize,
    },
    InvalidAxisValueNotAMultipleOf {
        arr_name: &'static str,
        axis_idx: usize,
        multiple: usize,
    },
    InvalidGeneric {
        msg: &'static str,
    },
    InvalidParameterEmptyArray {
        param_name: &'static str,
    },
    InvalidParameterValueEqual {
        param_name: &'static str,
        value: usize,
    },
    InvalidParameterValueGreater {
        param_name: &'static str,
        value: usize,
    },
    InvalidParameterValueLess {
        param_name: &'static str,
        value: usize,
    },
    InvalidParameterValueOutsideRange {
        param_name: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    InvalidSum {
        expected: f64,
        got: f64,
    },
    MismatchedArrayLengths {
        a_arr_name: &'static str,
        a_arr_len: usize,
        b_arr_name: &'static str,
        b_arr_len: usize,
    },
    MismatchedArrayShapes {
        a_arr_name: &'static str,
        a_shape: Vec<usize>,
        b_arr_name: &'static str,
        b_shape: Vec<usize>,
    },
}

impl fmt::Display for ImgalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImgalError::InvalidAxis { axis_idx, dim_len } => {
                write!(
                    f,
                    "Invalid axis, axis {} is out of bounds for dimension length {}.",
                    axis_idx, dim_len
                )
            }
            ImgalError::InvalidArrayLengthExpected {
                arr_name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Invalid array length, \"{}\" of length {} expected, but got {}.",
                    arr_name, expected, got
                )
            }
            ImgalError::InvalidAxisValueGreaterEqual {
                arr_name,
                axis_idx,
                value,
            } => {
                write!(
                    f,
                    "Invalid axis value, axis {} of \"{}\" can not be greater than or equal to {}.",
                    axis_idx, arr_name, value
                )
            }
            ImgalError::InvalidAxisValueNotAMultipleOf {
                arr_name,
                axis_idx,
                multiple,
            } => {
                write!(
                    f,
                    "Invalid axis value, axis {} of \"{}\" is not a multiple of {}.",
                    axis_idx, arr_name, multiple
                )
            }
            ImgalError::InvalidGeneric { msg } => {
                write!(f, "{}", msg)
            }
            ImgalError::InvalidParameterEmptyArray { param_name } => {
                write!(
                    f,
                    "Invalid array parameter, the array \"{}\" can not be empty.",
                    param_name
                )
            }
            ImgalError::InvalidParameterValueEqual { param_name, value } => {
                write!(
                    f,
                    "Invalid parameter value, the parameter \"{}\" can not equal {}.",
                    param_name, value
                )
            }
            ImgalError::InvalidParameterValueGreater { param_name, value } => {
                write!(
                    f,
                    "Invalid parameter value, the parameter \"{}\" can not be greater than {}.",
                    param_name, value
                )
            }
            ImgalError::InvalidParameterValueLess { param_name, value } => {
                write!(
                    f,
                    "Invalid parameter value, the parameter \"{}\" can not be less than {}.",
                    param_name, value
                )
            }
            ImgalError::InvalidParameterValueOutsideRange {
                param_name,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "Invalid parameter value, the parameter {} must be a value between {} and {} but got {}.",
                    param_name, min, max, value
                )
            }
            ImgalError::InvalidSum { expected, got } => {
                write!(f, "Invalid sum, expected {} but got {}.", expected, got)
            }
            ImgalError::MismatchedArrayLengths {
                a_arr_name,
                a_arr_len,
                b_arr_name,
                b_arr_len,
            } => {
                write!(
                    f,
                    "Mismatched array lengths, \"{}\" of length {} and \"{}\" of length {} do not match.",
                    a_arr_name, a_arr_len, b_arr_name, b_arr_len
                )
            }
            ImgalError::MismatchedArrayShapes {
                a_arr_name,
                a_shape,
                b_arr_name,
                b_shape,
            } => {
                write!(
                    f,
                    "Mismatched array shapes, array \"{}\" with shape {:?} and array \"{}\" with shape {:?} do not match.",
                    a_arr_name, a_shape, b_arr_name, b_shape
                )
            }
        }
    }
}

impl error::Error for ImgalError {}
