use std::iter::Sum;

use super::DenseMat;

use crate::matrix::mat_shape::MatShape;

impl<T: crate::FloatCore + Sum> DenseMat<T> {
    pub fn mat_norm(&self) -> Option<T> {
        assert!(self.is_square());

        (0..self.ny)
            .map(|iy| (0..self.nx).map(|ix| self[(ix, iy)].abs()).sum())
            .reduce(T::max)
    }

    pub fn condition_number(&self) -> Option<T> {
        Some(self.mat_norm()? * self.inverse().mat_norm()?)
    }

    pub fn is_strictly_diagonally_dominant(&self) -> bool {
        assert!(self.is_square());

        (0..self.ny).all(|iy| {
            (0..self.nx)
                .filter(|ix| *ix != iy)
                .map(|ix| self[(ix, iy)].abs())
                .sum::<T>()
                < self[(iy, iy)].abs()
        })
    }

    pub fn contains_nan(&self) -> bool {
        self.data
            .iter()
            .filter(|x| crate::FloatCore::is_nan(**x))
            .count()
            > 0
    }
}
