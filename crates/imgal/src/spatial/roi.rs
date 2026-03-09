use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayBase, AsArray, Axis, Dimension, ViewRepr};
use rayon::prelude::*;

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// Create a ROI point cloud map from an n-dimensional label image.
///
/// # Description
///
/// Creates a region of interest (ROI) "cloud" map from an n-dimensional label
/// image. For a given input image each label is converted into a 2D array
/// representing a point cloud with shape `(p, D)`, where `p` and `D` are the
/// number of points and dimensions respectively. Each label's point cloud is
/// stored with it's associated key (*i.e.* label ID) in the output `HashMap`.
///
/// # Arguments
///
/// * `labels`: The n-dimensional label image.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `HashMap<u64, Array2<usize>>`: A ROI `HashMap` where the keys are the ROI
///   label IDs and values are the ROI point clouds.
pub fn roi_cloud_map<'a, A, D>(labels: A, parallel: bool) -> HashMap<u64, Array2<usize>>
where
    A: AsArray<'a, u64, D>,
    D: Dimension,
{
    let data: ArrayBase<ViewRepr<&'a u64>, D> = labels.into();
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
                HashMap::new,
                |mut map: HashMap<u64, Vec<Vec<usize>>>, (p, &v)| {
                    map.entry(v).or_default().push(p.as_array_view().to_vec());
                    map
                },
            )
            .reduce(HashMap::new, |mut map_a, map_b| {
                map_b.into_iter().for_each(|(k, mut v)| {
                    map_a.entry(k).or_insert_with(Vec::new).append(&mut v);
                });
                map_a
            });
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
                    .or_default()
                    .push(p.as_array_view().to_vec());
            });
        cloud_map
            .into_iter()
            .map(|(k, v)| vec_to_arr(k, v))
            .collect()
    }
}

/// Create a ROI data map from n-dimensional data and a label image.
///
/// # Description
///
/// Creates a region of interest (ROI) "data" map from input n-dimensional data
/// and label images. For a given `data` and `labels` image pair, each
/// coordinate within every label in the label image is used to query the
/// image data. Each label's associated raw data is stored as a 1D array with
/// the label's key (*i.e.* label ID) in the output `HashMap`.
///
/// # Arguments
///
/// * `data`: The input n-dimensional image data.
/// * `labels`: The corresponding n-dimensional label image for `data`.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(HashMap<u64, Array1<T>>)`: A ROI `HashMap` where the keys are the ROI
///   label IDs and the values are 1D arrays containing raw values from the ROI.
/// * `Err(ImgalError)`: If `data.shape() != labels.shape()`.
pub fn roi_data_map<'a, T, A, B, D>(
    data: A,
    labels: B,
    parallel: bool,
) -> Result<HashMap<u64, Array1<T>>, ImgalError>
where
    A: AsArray<'a, T, D>,
    B: AsArray<'a, u64, D>,
    D: Dimension,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let labels: ArrayBase<ViewRepr<&'a u64>, D> = labels.into();
    if data.shape() != labels.shape() {
        return Err(ImgalError::MismatchedArrayShapes {
            a_arr_name: "data",
            a_shape: data.shape().to_vec(),
            b_arr_name: "labels",
            b_shape: labels.shape().to_vec(),
        });
    }
    let data = data.into_dyn();
    let rcm = roi_cloud_map(labels, parallel);
    let mut rdm: HashMap<u64, Array1<T>> = HashMap::new();
    rcm.iter().for_each(|(&k, c)| {
        let cloud_lns = c.lanes(Axis(1));
        let cloud_data = cloud_lns
            .into_iter()
            .map(|l| {
                // safe to unwrap here because the ROI point cloud map is made from
                // the given labels array (not the caller) which must have matching
                // shape as the given data array
                match l.as_slice() {
                    Some(s) => *data.get(s).unwrap(),
                    None => {
                        let coords = l.to_vec();
                        *data.get(coords.as_slice()).unwrap()
                    }
                }
            })
            .collect::<Vec<T>>();
        rdm.insert(k, Array1::from_vec(cloud_data));
    });
    Ok(rdm)
}
