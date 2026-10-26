use pyo3::prelude::*;

/// Register a module in `sys.modules` under its full dotted path.
pub(crate) fn register_in_sys_modules(m: &Bound<'_, PyModule>, path: &str) -> PyResult<()> {
    m.py().import("sys")?.getattr("modules")?.set_item(path, m)
}
