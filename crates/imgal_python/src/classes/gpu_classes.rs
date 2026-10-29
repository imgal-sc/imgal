use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use imgal::gpu::GpuTensor;

/// Tensor dtype variants.
///
/// PyO3 does not allow generic <T> with structs, so this enum lets us provide
/// provide one GPUTensor that accepts multiple data types.
enum TensorDtype {
    U32(GpuTensor<u32>),
    U64(GpuTensor<u64>),
    I32(GpuTensor<i32>),
    I64(GpuTensor<i64>),
    F32(GpuTensor<f32>),
    F64(GpuTensor<f64>),
}

/// A tensor with a memory allocation on the GPU.
///
/// The `GPUTensor` represents array data (*i.e.* a tensor) on an initialized
/// GPU. An active `GPUTensor` will always have *some* GPU memory allocation
/// and a shape.
#[pyclass(name = "GPUTensor")]
pub struct PyGpuTensor(TensorDtype);

#[pymethods]
impl PyGpuTensor {
    /// Create a new empty GPU tensor.
    ///
    /// Creates a new empty GPU tensor with a size defined by `shape`.
    ///
    /// Args:
    ///     The shape of the empty tensor.
    ///
    /// Returns:
    ///     An initialized empty GPU tensor.
    #[staticmethod]
    pub fn empty(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        match dtype {
            "uint32" => Ok(Self(TensorDtype::U32(GpuTensor::empty(&shape)))),
            "uint64" => Ok(Self(TensorDtype::U64(GpuTensor::empty(&shape)))),
            "int32" => Ok(Self(TensorDtype::I32(GpuTensor::empty(&shape)))),
            "int64" => Ok(Self(TensorDtype::I64(GpuTensor::empty(&shape)))),
            "float32" => Ok(Self(TensorDtype::F32(GpuTensor::empty(&shape)))),
            "float64" => Ok(Self(TensorDtype::F64(GpuTensor::empty(&shape)))),
            _ => Err(PyErr::new::<PyTypeError, _>(
                "Unsupported tensor dtype. Supported types are u32, u64, i32, i64, f32 and f64.",
            )),
        }
    }

    /// Create a new empty GPU tensor of size one.
    ///
    /// Creates a new empty GPU tensor of size one. A size one tensor is useful
    /// for functions that reduce an input tensor into one value.
    ///
    /// Returns:
    ///     An initialized empty GPU tensor with a size of one.
    #[staticmethod]
    pub fn empty_one(dtype: &str) -> PyResult<Self> {
        match dtype {
            "uint32" => Ok(Self(TensorDtype::U32(GpuTensor::empty_one()))),
            "uint64" => Ok(Self(TensorDtype::U64(GpuTensor::empty_one()))),
            "int32" => Ok(Self(TensorDtype::I32(GpuTensor::empty_one()))),
            "int64" => Ok(Self(TensorDtype::I64(GpuTensor::empty_one()))),
            "float32" => Ok(Self(TensorDtype::F32(GpuTensor::empty_one()))),
            "float64" => Ok(Self(TensorDtype::F64(GpuTensor::empty_one()))),
            _ => Err(PyErr::new::<PyTypeError, _>(
                "Unsupported tensor dtype. Supported types are u32, u64, i32, i64, f32 and f64.",
            )),
        }
    }

    /// Create a new GPU tensor with input data.
    ///
    /// Creates a new GPU tensor from the input *n*-dimensional image.
    ///
    /// Args:
    ///     The input *n*-dimensional image.
    ///
    /// Returns:
    ///     An initialized GPU tensor with the input data.
    #[staticmethod]
    pub fn new<'py>(data: Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u32>>() {
            Ok(Self(TensorDtype::U32(GpuTensor::new(arr.as_array()))))
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
            Ok(Self(TensorDtype::U64(GpuTensor::new(arr.as_array()))))
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i32>>() {
            Ok(Self(TensorDtype::I32(GpuTensor::new(arr.as_array()))))
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
            Ok(Self(TensorDtype::I64(GpuTensor::new(arr.as_array()))))
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
            Ok(Self(TensorDtype::F32(GpuTensor::new(arr.as_array()))))
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
            Ok(Self(TensorDtype::F64(GpuTensor::new(arr.as_array()))))
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "Unsupported array dtype. Supported array dtypes are u32, u64, i32, i64, f32 and f64.",
            ))
        }
    }

    /// Read data from the GPU tensor.
    ///
    /// Reads *n*-dimensional data from the tensor's reserved GPU memory.
    ///
    /// Returns:
    ///     The tensor data.
    pub fn read<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.0 {
            TensorDtype::U32(t) => t.read().into_pyarray(py).into_any(),
            TensorDtype::U64(t) => t.read().into_pyarray(py).into_any(),
            TensorDtype::I32(t) => t.read().into_pyarray(py).into_any(),
            TensorDtype::I64(t) => t.read().into_pyarray(py).into_any(),
            TensorDtype::F32(t) => t.read().into_pyarray(py).into_any(),
            TensorDtype::F64(t) => t.read().into_pyarray(py).into_any(),
        }
    }

    /// Get the shape of the tensor.
    ///
    /// Gets the shape of the tensor. Once a tensor is initialized the shape is
    /// immutable.
    ///
    /// Returns:
    ///     The shape of the tensor.
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        match &self.0 {
            TensorDtype::U32(t) => t.shape.clone(),
            TensorDtype::U64(t) => t.shape.clone(),
            TensorDtype::I32(t) => t.shape.clone(),
            TensorDtype::I64(t) => t.shape.clone(),
            TensorDtype::F32(t) => t.shape.clone(),
            TensorDtype::F64(t) => t.shape.clone(),
        }
    }

    /// Get the size of the tensor.
    ///
    /// Gets the sieze of the tensor. Once a tensor is initialized the shape is
    /// immutable.
    ///
    /// Returns:
    ///     The size of the tensor.
    #[getter]
    pub fn size(&self) -> usize {
        match &self.0 {
            TensorDtype::U32(t) => t.size,
            TensorDtype::U64(t) => t.size,
            TensorDtype::I32(t) => t.size,
            TensorDtype::I64(t) => t.size,
            TensorDtype::F32(t) => t.size,
            TensorDtype::F64(t) => t.size,
        }
    }

    /// Write data into the GPU tensor.
    ///
    /// Writes *n*-dimensional data into the tensor's reserved GPU memory.
    ///
    /// Args:
    ///     The input *n*-dimensional image.
    pub fn write<'py>(&self, data: Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u32>>() {
            match &self.0 {
                TensorDtype::U32(t) => Ok(t.write(arr.as_array())),
                _ => Err(PyErr::new::<PyTypeError, _>(
                    "Failed to write uint32 data to the tensor.",
                )),
            }
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<u64>>() {
            match &self.0 {
                TensorDtype::U64(t) => Ok(t.write(arr.as_array())),
                _ => Err(PyErr::new::<PyTypeError, _>(
                    "Failed to write uint64 data to the tensor.",
                )),
            }
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i32>>() {
            match &self.0 {
                TensorDtype::I32(t) => Ok(t.write(arr.as_array())),
                _ => Err(PyErr::new::<PyTypeError, _>(
                    "Failed to write int32 data to the tensor.",
                )),
            }
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<i64>>() {
            match &self.0 {
                TensorDtype::I64(t) => Ok(t.write(arr.as_array())),
                _ => Err(PyErr::new::<PyTypeError, _>(
                    "Failed to write int64 data to the tensor.",
                )),
            }
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f32>>() {
            match &self.0 {
                TensorDtype::F32(t) => Ok(t.write(arr.as_array())),
                _ => Err(PyErr::new::<PyTypeError, _>(
                    "Failed to write float32 data to the tensor.",
                )),
            }
        } else if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<f64>>() {
            match &self.0 {
                TensorDtype::F64(t) => Ok(t.write(arr.as_array())),
                _ => Err(PyErr::new::<PyTypeError, _>(
                    "Failed to write float64 data to the tensor.",
                )),
            }
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "Unsupported array dtype. Supported array dtypes are u32, u64, i32, i64, f32 and f64.",
            ))
        }
    }
}
