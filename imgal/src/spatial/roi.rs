use std::collections::HashMap;

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
///
/// # Returns
///
/// * `HashMap<u16, Array2<usize>>`: A ROI HashMap where the keys are the ROI
///   labels and values are the ROI point clouds.
pub fn roi_map<'a, A, D>(data: A) -> HashMap<u64, Array2<usize>>
where
    A: AsArray<'a, u64, D>,
    D: Dimension,
{
    let data: ArrayBase<ViewRepr<&'a u64>, D> = data.into();
    let mut roi_map: HashMap<u64, Vec<Vec<usize>>> = HashMap::new();
    data.view()
        .into_dyn()
        .indexed_iter()
        .filter(|&(_, &v)| v != 0)
        .for_each(|(p, &v)| match roi_map.get_mut(&v) {
            Some(cloud) => {
                cloud.push(p.as_array_view().to_vec());
            }
            None => {
                let mut cloud: Vec<Vec<usize>> = Vec::new();
                cloud.push(p.as_array_view().to_vec());
                roi_map.insert(v, cloud);
            }
        });

    roi_map
        .into_iter()
        .map(|(k, v)| {
            let arr =
                Array2::from_shape_vec((v.len(), v[0].len()), v.into_iter().flatten().collect())
                    .expect("Failed to reshape ROI point cloud into an Array2<usize>.");
            (k, arr)
        })
        .collect()
}
