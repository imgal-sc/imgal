use std::cmp::Ordering;

use ndarray::{Array2, ArrayBase, ArrayView2, AsArray, Axis, Ix2, ViewRepr};

use crate::error::ImgalError;
use crate::traits::numeric::AsNumeric;

/// An immutable K-d tree for fast spatial queries for n-dimensional points.
///
/// The K-d tree itself does not *own* its source data but instead uses a view.
/// This design ensures that imgal's K-d trees are *immutable* once constructed
/// and are intended for lookups only. The `cloud` view (*i.e.* the
/// *n*-dimensional point cloud) points in *D* dimensions with shape `(p, D)`,
/// where `p` is the point and `D` is the dimension/axis of that point.
pub struct KDTree<'a, T> {
    /// A view into a point cloud array with shape `(p, D)`.
    pub cloud: ArrayView2<'a, T>,
    /// The K-d tree node vector that each `Node` indexes into.
    pub nodes: Vec<Node>,
    /// The root of the K-d tree.
    pub root: Option<usize>,
}

/// A K-d-tree node for an immutable K-d tree.
///
/// KD-trees are constructed with `Node`s. These `Nodes` are stored in a
/// `Vec<Node>` and the `left` and `right` fields store indices into the `Node`
/// vector. The axis the split occurs at is stored in `split_axis` and the index
/// into the source array is stored in the `point_index` field.
pub struct Node {
    /// The axis this node was split on.
    pub split_axis: usize,
    /// The node's current point index into the K-d tree's associated point
    /// cloud.
    pub point_index: usize,
    /// The index into the "left" branch relative to this node.
    pub left: Option<usize>,
    /// The index into the "right" branch relative to this node.
    pub right: Option<usize>,
}

impl<'a, T> KDTree<'a, T>
where
    T: AsNumeric,
{
    /// Create a new K-d tree from an *n*-dimensional point cloud.
    ///
    /// # Description
    ///
    /// Creates a new K-d t ree from an *n*-dimensional point cloud with an
    /// array shape of `(p, D)`, where `p` is the point and `D` is the
    /// dimension/axis of that point. The `KDTree` does not own the point cloud
    /// data, but instead owns an array of `Nodes` that store indices into
    /// the source point cloud.
    ///
    /// # Arguments
    ///
    /// * `cloud`: An array view into a point cloud with shape `(p, D)`.
    ///
    /// # Returns
    ///
    /// * `KDTree<'a, T>`: A K-d tree with radial searching of either point
    ///   indices or coordinates.
    pub fn build<A>(cloud: A) -> Self
    where
        A: AsArray<'a, T, Ix2>,
    {
        let view: ArrayBase<ViewRepr<&'a T>, Ix2> = cloud.into();
        let mut tree = Self {
            cloud: view,
            nodes: Vec::new(),
            root: None,
        };
        let total_points = view.dim().0;
        let indices: Vec<usize> = (0..total_points).collect();
        tree.root = tree.recursive_build(&indices, 0);

        tree
    }

    /// Search the K-d tree for all point coordinates within a given radius.
    ///
    /// # Description
    ///
    /// Performs a radial search on the K-d tree, returning the coordinates of
    /// all points whose Euclidean distance from the `query` point is less than
    /// or equal to `radius`.
    ///
    /// # Arguments
    ///
    /// * `query`: A slice representing the query point. The query point length
    ///   must match the dimension length of the point cloud.
    /// * `radius`: The radius around the query point to search.
    ///
    /// # Returns
    ///
    /// * `Ok(Array2<T>)`: The point coordinates of all neighboring points to the
    ///   `query` within the `radius`. The returned array has shape `(p, D)`,
    ///   where `p` is the point and `D` is the dimension/axis of that point.
    /// * `Err(ImgalError)`: If `query.len() != self.cloud.dim().1`.
    pub fn search_for_coords(&self, query: &[T], radius: f64) -> Result<Array2<T>, ImgalError> {
        let q_dims = query.len();
        let c_dims = self.cloud.dim().1;
        if q_dims != c_dims {
            return Err(ImgalError::MismatchedArrayLengths {
                a_arr_name: "query",
                a_arr_len: q_dims,
                b_arr_name: "cloud array shape",
                b_arr_len: c_dims,
            });
        }
        let coord_indices = self.search_for_indices(query, radius).unwrap();

        Ok(self.cloud.select(Axis(0), &coord_indices))
    }

    /// Search the K-d tree for all point indices within the given radius.
    ///
    /// # Description
    ///
    /// Performs a radial search on the K-d tree, returning the indices of all
    /// points whose Euclidean distance from the `query` point is less than or
    /// equal to `radius`.
    ///
    /// # Arguments
    ///
    /// * `query`: A slice representing the query point. The query point length
    ///   must match the dimension length of the point cloud.
    /// * `radius`: The radius around the query point to search.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<usize>)`: The point indices of all neighboring points to the query
    ///   within the `radius`.
    /// * `Err(ImgalError)`: If `query.len() != self.cloud.dim().1`.
    pub fn search_for_indices(&self, query: &[T], radius: f64) -> Result<Vec<usize>, ImgalError> {
        let q_dims = query.len();
        let c_dims = self.cloud.dim().1;
        if q_dims != c_dims {
            return Err(ImgalError::MismatchedArrayLengths {
                a_arr_name: "query",
                a_arr_len: q_dims,
                b_arr_name: "cloud array shape",
                b_arr_len: c_dims,
            });
        }
        let radius_sq = radius.powi(2);
        let mut results: Vec<usize> = Vec::new();

        // begin recursive searching only if the tree is not empty
        if let Some(root) = self.root {
            self.recursive_search(root, q_dims, query, radius_sq, &mut results);
        }

        Ok(results)
    }

    /// Recursively build the K-d tree.
    fn recursive_build(&mut self, indices: &[usize], depth: usize) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }
        let n_dims = self.cloud.dim().1;
        let split_axis = depth % n_dims;
        let mut inds_sorted = indices.to_vec();
        // sort the indices associated with the points, no need to mutate the data
        inds_sorted.sort_by(|&a, &b| {
            self.cloud[[a, split_axis]]
                .partial_cmp(&self.cloud[[b, split_axis]])
                .unwrap_or(Ordering::Less)
        });
        let median = inds_sorted.len() / 2;
        let point_index = inds_sorted[median];
        // construct the left and right sub trees
        let left = self.recursive_build(&inds_sorted[..median], depth + 1);
        let right = self.recursive_build(&inds_sorted[median + 1..], depth + 1);
        // create a new Node and return this Node's index
        let node_index = self.nodes.len();
        self.nodes
            .push(Node::new(split_axis, point_index, left, right));

        Some(node_index)
    }

    /// Recursively search the K-d tree.
    fn recursive_search(
        &self,
        node_index: usize,
        n_dims: usize,
        query: &[T],
        radius_sq: f64,
        results: &mut Vec<usize>,
    ) {
        // get the current node's distance from the query point and add this
        // point if we're within the radius squared
        let node = &self.nodes[node_index];
        let mut node_point: Vec<T> = Vec::with_capacity(n_dims);
        (0..n_dims).for_each(|k| {
            node_point.push(self.cloud[[node.point_index, k]]);
        });
        let node_dist_sq = node_point
            .iter()
            .zip(query.iter())
            .fold(0.0, |acc, (&n, &q)| {
                let d = n.to_f64() - q.to_f64();
                acc + (d * d)
            });
        if node_dist_sq <= radius_sq {
            results.push(node.point_index);
        }

        // decide the transveral order and recurse into the near side and far
        // side (only if needed)
        let ax = node.split_axis;
        let diff = query[ax].to_f64() - node_point[ax].to_f64();
        let (near, far) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        if let Some(child) = near {
            self.recursive_search(child, n_dims, query, radius_sq, results);
        }
        if diff.powi(2) <= radius_sq {
            if let Some(child) = far {
                self.recursive_search(child, n_dims, query, radius_sq, results);
            }
        }
    }
}

impl Node {
    /// Creates a new K-d tree node.
    pub fn new(
        split_axis: usize,
        point_index: usize,
        left: Option<usize>,
        right: Option<usize>,
    ) -> Self {
        Self {
            split_axis,
            point_index,
            left,
            right,
        }
    }
}
