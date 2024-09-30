use std::iter::Sum;
use std::ops::{Add, Sub};

use approx::AbsDiffEq;

use crate::matrix::mat_shape::MatShape;
use crate::matrix::mat_traits::MatMulVec;

use super::DenseMat;

impl<T> DenseMat<T> {
    pub fn swap_element(&mut self, pos_a: (usize, usize), pos_b: (usize, usize)) {
        let pos_a = self.index_2d_to_1d(pos_a.0, pos_a.1);
        let pos_b = self.index_2d_to_1d(pos_b.0, pos_b.1);
        self.data.swap(pos_a, pos_b)
    }

    pub fn swap_iy(&mut self, iy_0: usize, iy_1: usize) {
        (0..self.nx).for_each(|ix| self.swap_element((ix, iy_0), (ix, iy_1)))
    }

    pub fn swap_ix(&mut self, ix_0: usize, ix_1: usize) {
        (0..self.ny).for_each(|iy| self.swap_element((ix_0, iy), (ix_1, iy)))
    }
}

impl<T: Clone> DenseMat<T> {
    pub fn transpose(&self) -> Self {
        let mut m = self.clone();
        m.nx = self.ny;
        m.ny = self.nx;
        (0..self.nx)
            .for_each(|ix| (0..self.ny).for_each(|iy| m[(iy, ix)] = self[(ix, iy)].clone()));
        m
    }
}

impl<T: crate::FloatCore> DenseMat<T> {
    pub fn add_mat(&mut self, other: &Self) {
        assert!(self.shape_eq(other));

        (0..self.nx).for_each(|ix| {
            (0..self.ny).for_each(|iy| self[(ix, iy)] = self[(ix, iy)] + other[(ix, iy)])
        });
    }

    pub fn sub_mat(&mut self, other: &Self) {
        assert!(self.shape_eq(other));

        (0..self.nx).for_each(|ix| {
            (0..self.ny).for_each(|iy| self[(ix, iy)] = self[(ix, iy)] - other[(ix, iy)])
        });
    }
}

impl<T: crate::FloatCore> Add for &DenseMat<T> {
    type Output = DenseMat<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut m = self.clone();
        m.add_mat(rhs);
        m
    }
}

impl<T: crate::FloatCore> Sub for &DenseMat<T> {
    type Output = DenseMat<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut m = self.clone();
        m.sub_mat(rhs);
        m
    }
}

impl<T: crate::FloatCore + AbsDiffEq<Epsilon = T>> AbsDiffEq for DenseMat<T> {
    type Epsilon = T;

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.nx == other.nx
            && self.ny == other.ny
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }

    fn default_epsilon() -> Self::Epsilon {
        <T as AbsDiffEq>::default_epsilon()
    }
}

mod mat_mul {
    use super::*;

    impl<T: crate::FloatCore + Sum> MatMulVec<T> for DenseMat<T> {
        fn mul_vec(&self, b: &[T]) -> Vec<T> {
            assert!(self.nx == b.len());

            (0..self.ny)
                .map(|iy| (0..self.nx).map(|ix| self[(ix, iy)] * b[ix]).sum())
                .collect::<Vec<T>>()
        }
    }

    impl<T: crate::FloatCore + Sum> DenseMat<T> {
        pub fn mul_mat_local(&self, other: &Self, store: &mut Self) {
            assert!(self.nx == other.ny);

            let nx = other.nx;
            let ny = self.ny;

            assert!(store.nx == nx && store.ny == ny);

            for iy in 0..ny {
                for ix in 0..nx {
                    store[(ix, iy)] = (0..self.nx).map(|j| self[(j, iy)] * other[(ix, j)]).sum();
                }
            }
        }

        pub fn mul_mat(&self, other: &Self) -> Self {
            let mut mat = Self::zeros(other.nx, self.ny);
            self.mul_mat_local(other, &mut mat);
            mat
        }
    }
}
