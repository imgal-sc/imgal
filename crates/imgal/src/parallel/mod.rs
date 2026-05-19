/// Do either sequential or parallel work
#[macro_export]
macro_rules! par {
    ($threads:expr, sequential: $seq_fn:expr, parallel: $par_fn:expr) => {{
        let resolve_threads = |req: usize| -> usize {
            let free_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            match req {
                0 => free_threads,
                n => n.min(free_threads),
            }
        };
        let get_pool = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        };
        match resolve_threads($threads) {
            1 => $seq_fn,
            n => get_pool(n).install(|| $par_fn),
        }
    }};
}
