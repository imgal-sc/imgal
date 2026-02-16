use std::collections::HashMap;

use rayon::prelude::*;

use ndarray::{Array2, ArrayBase, AsArray, Dimension, ViewRepr};

/// Create a ROI map from an n-dimensional label image.
///
/// # Description
///
/// Creates a region of interest (ROI) map from an n-dimensional label image.
/// For a given input image each label is converted into a 2D point cloud with
/// shape `(p, D)`, where `p` and `D` are the number of points and dimensions
/// respectively.
///
/// # Arguments
///
/// * `data`: An n-dimensional label image of type u16.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `HashMap<u16, Array2<usize>>`: A ROI HashMap where the keys are the ROI
///   labels and values are the ROI point clouds.
pub fn roi_map<'a, A, D>(data: A, parallel: bool) -> HashMap<u64, Array2<usize>>
where
    A: AsArray<'a, u64, D>,
    D: Dimension,
{
    let data: ArrayBase<ViewRepr<&'a u64>, D> = data.into();
    let vec_to_arr = |k: u64, v: Vec<Vec<usize>>| {
        let arr = Array2::from_shape_vec((v.len(), v[0].len()), v.into_iter().flatten().collect())
            .expect("Failed to reshape ROI point cloud into an Array2<usize>.");
        (k, arr)
    };
    if parallel {
        let cloud_map = data
            .view()
            .into_dyn()
            .indexed_iter()
            .par_bridge()
            .filter(|&(_, &v)| v != 0)
            .fold(
                || HashMap::new(),
                |mut map: HashMap<u64, Vec<Vec<usize>>>, (p, &v)| {
                    map.entry(v)
                        .or_insert_with(Vec::new)
                        .push(p.as_array_view().to_vec());
                    map
                },
            )
            .reduce(
                || HashMap::new(),
                |mut map_a, map_b| {
                    map_b.into_iter().for_each(|(k, mut v)| {
                        map_a.entry(k).or_insert_with(Vec::new).append(&mut v);
                    });
                    map_a
                },
            );
        cloud_map
            .into_par_iter()
            .map(|(k, v)| vec_to_arr(k, v))
            .collect()
    } else {
        let mut cloud_map: HashMap<u64, Vec<Vec<usize>>> = HashMap::new();
        data.view()
            .into_dyn()
            .indexed_iter()
            .filter(|&(_, &v)| v != 0)
            .for_each(|(p, &v)| {
                cloud_map
                    .entry(v)
                    .or_insert_with(Vec::new)
                    .push(p.as_array_view().to_vec());
            });
        cloud_map
            .into_iter()
            .map(|(k, v)| vec_to_arr(k, v))
            .collect()
    }
}
