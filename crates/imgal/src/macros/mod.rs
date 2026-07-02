/// Dispatch work to either a sequential iterator or parallel execution path
/// with the requested thread count.
///
/// # Example
///
/// ```ignore
/// let threads = Some(2);
/// let data: Vec<usize> = par!(
///      threads,
///      seq_exp: (0..10).map(|i| i * 2).collect(),
///      par_exp: (0..10).into_par_iter().map(|i| i * 2).collect()
///  );
/// ```
macro_rules! par {
    ($threads:expr, seq_exp: $seq:expr, par_exp: $par:expr) => {{
        match $threads.unwrap_or(1) {
            1 => $seq,
            0 => crate::macros::get_pool(usize::MAX).install(|| $par),
            n => crate::macros::get_pool(n).install(|| $par),
        }
    }};
}

/// Helper function to construct thread pools for the par! macro.
pub fn get_pool(n: usize) -> rayon::ThreadPool {
    let max = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n.min(max))
        .build()
        .unwrap()
}
