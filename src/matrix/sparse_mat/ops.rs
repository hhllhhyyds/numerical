use std::collections::BTreeMap;
use std::iter::Sum;
use std::ops::{Add, Sub};

use approx::AbsDiffEq;

use crate::matrix::mat_shape::MatShape;
use crate::matrix::mat_traits::MatMulVec;
use crate::FloatCore;

use super::SparseMat;

impl<T: Clone> SparseMat<T> {
    pub fn transpose(&self) -> Self {
        Self::new(
            self.ny,
            self.nx,
            self.data
                .iter()
                .map(|((ix, iy), val)| (*iy, *ix, val.clone())),
        )
    }
}

impl<T: FloatCore + Sum> MatMulVec<T> for SparseMat<T> {
    fn mul_vec(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());

        let mut v = vec![T::zero(); self.ny];
        for (&(ix, iy), &val) in self.data.iter() {
            v[iy] = v[iy] + val * b[ix];
        }

        v
    }
}

impl<T: FloatCore> SparseMat<T> {
    pub fn mul_diagonal(&self, diag: &[T]) -> Self {
        assert!(self.nx == diag.len());

        let mut mat = Self {
            nx: self.nx,
            ny: self.ny,
            data: BTreeMap::default(),
        };

        for (&(ix, iy), &val) in self.data.iter() {
            mat.set((ix, iy), val * diag[ix])
        }

        mat
    }
}

impl<T: FloatCore> SparseMat<T> {
    pub fn add_mat(&mut self, other: &Self) {
        assert!(self.shape_eq(other));

        for (&(ix, iy), &val) in other.data.iter() {
            let sum = *self.get((ix, iy)).unwrap_or(&T::zero()) + val;
            self.set((ix, iy), sum);
        }
    }

    pub fn sub_mat(&mut self, other: &Self) {
        assert!(self.shape_eq(other));

        for (&(ix, iy), &val) in other.data.iter() {
            let sum = *self.get((ix, iy)).unwrap_or(&T::zero()) - val;
            self.set((ix, iy), sum);
        }
    }
}

impl<T: crate::FloatCore> Add for &SparseMat<T> {
    type Output = SparseMat<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut m = self.clone();
        m.add_mat(rhs);
        m
    }
}

impl<T: crate::FloatCore> Sub for &SparseMat<T> {
    type Output = SparseMat<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut m = self.clone();
        m.sub_mat(rhs);
        m
    }
}

impl<T: crate::FloatCore + AbsDiffEq<Epsilon = T>> AbsDiffEq for SparseMat<T> {
    type Epsilon = T;

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.nx == other.nx
            && self.ny == other.ny
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.0 == b.0 && a.1.abs_diff_eq(b.1, epsilon))
    }

    fn default_epsilon() -> Self::Epsilon {
        <T as AbsDiffEq>::default_epsilon()
    }
}
