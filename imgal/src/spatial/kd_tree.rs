use std::cmp::Ordering;

use ndarray::{Array2, ArrayBase, ArrayView2, AsArray, Axis, Ix1, Ix2, ViewRepr};

use crate::traits::numeric::AsNumeric;

/// An immutable K-d tree for fast spatial queries for n-dimensional points.
///
/// The KD-tree itself does not *own* its source data but instead uses a view.
/// This design ensures that imgal's KD-trees are *immutable* once constructed
/// and are intended for lookups only. The `cloud` view (*i.e.* the
/// *n*-dimensional point cloud) points in *k* dimensions with shape `(p, k)`,
/// where `p` is the point and `k` is the dimension/axis of that point.
pub struct KDTree<'a, T> {
    pub cloud: ArrayView2<'a, T>,
    pub nodes: Vec<Node>,
    pub root: Option<usize>,
}

/// A K-d-tree node for an immutable K-d tree.
///
/// KD-trees are constructed with `Node`s. These `Nodes` are stored in a
/// `Vec<Node>` and the `left` and `right` fields store indices into the `Node`
/// vector. The axis the split occurs at is stored in `split_axis` and the index
/// into the source array is stored in the `point_index` field.
pub struct Node {
    pub split_axis: usize,
    pub point_index: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl<'a, T> KDTree<'a, T>
where
    T: AsNumeric,
{
    /// Creates a new K-d tree from an *n*-dimensional point cloud.
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

    /// Search the K-d tree for all points with in the given radius.
    pub fn search<A>(&self, query: A, radius: T) -> Array2<T>
    where
        A: AsArray<'a, T, Ix1>,
    {
        let view: ArrayBase<ViewRepr<&'a T>, Ix1> = query.into();
        let query = view.as_slice().unwrap();
        let n_dims = query.len();
        let radius_sq = radius * radius;
        let mut results: Vec<usize> = Vec::new();
        // TODO: ensure query element length (i.e. dimensions) is equal to n_dims

        // begin recursive searching only if the tree is not empty
        if let Some(root) = self.root {
            self.recursive_search(root, n_dims, query, radius_sq, &mut results);
        }

        // select the result indices from the point cloud
        self.cloud.select(Axis(0), &results)
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
        radius_sq: T,
        results: &mut Vec<usize>,
    ) {
        // get the current node's coordinates
        let node = &self.nodes[node_index];
        let mut node_point: Vec<T> = Vec::with_capacity(n_dims);
        (0..n_dims).for_each(|k| {
            node_point.push(self.cloud[[node.point_index, k]]);
        });

        // compute the current node's distance from the query
        let node_dist_sq =
            node_point
                .iter()
                .zip(query.iter())
                .fold(T::default(), |acc, (&n, &q)| {
                    let d = n - q;
                    acc + (d * d)
                });

        // add this node to results if it's within the specified radius
        if node_dist_sq <= radius_sq {
            results.push(node.point_index);
        }

        // decide the transveral order and recurse into the near side and far
        // side (only if needed)
        let ax = node.split_axis;
        let diff = query[ax] - node_point[ax];
        let (near, far) = if diff <= T::default() {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        if let Some(child) = near {
            self.recursive_search(child, n_dims, query, radius_sq, results);
        }
        if diff * diff <= radius_sq {
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
