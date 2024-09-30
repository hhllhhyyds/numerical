use crate::matrix::mat_shape::MatShape;

use super::DenseMat;

impl<T> DenseMat<T> {
    pub fn from_vec(nx: usize, ny: usize, data: Vec<T>) -> Self {
        assert!(data.len() == nx * ny);
        Self { nx, ny, data }
    }
}

impl<T: Clone> DenseMat<T> {
    pub fn from_slice(nx: usize, ny: usize, data: &[T]) -> Self {
        Self::from_vec(nx, ny, data.to_vec())
    }

    pub fn diagonal(&self) -> Vec<T> {
        assert!(self.is_square());
        (0..self.nx).map(|i| self[(i, i)].clone()).collect()
    }
}

impl<T: crate::FloatCore> DenseMat<T> {
    pub fn zeros(nx: usize, ny: usize) -> Self {
        Self::from_vec(nx, ny, vec![T::zero(); nx * ny])
    }

    pub fn from_diagonal(diag: &[T]) -> Self {
        let mut mat = Self::zeros(diag.len(), diag.len());
        (0..diag.len()).for_each(|i| mat[(i, i)] = diag[i]);
        mat
    }

    pub fn identity(n: usize) -> Self {
        Self::from_diagonal(&vec![T::one(); n])
    }
}
