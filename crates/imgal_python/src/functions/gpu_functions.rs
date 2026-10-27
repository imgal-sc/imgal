use pyo3::prelude::*;

use imgal::gpu::init_gpu;

/// Initialize the GPU and cache the compute client.
///
/// Initializes the GPU and caches the compute client in a OnceLock. The client
/// enables data transfer to and from the GPU.
#[pyfunction]
#[pyo3(name = "init_gpu")]
pub fn gpu_init_gpu() {
    init_gpu();
}
