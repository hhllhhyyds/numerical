use std::{
    iter::Sum,
    ops::{Index, IndexMut},
};

use crate::FloatCore;

#[derive(Debug, Clone)]
pub struct DenseMat<T: FloatCore> {
    nx: usize,
    ny: usize,
    data: Vec<T>,
}

impl<T: FloatCore + Sum> DenseMat<T> {
    pub fn mul_vec(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());

        (0..self.ny)
            .map(|iy| (0..self.nx).map(|ix| self[(ix, iy)] * b[ix]).sum())
            .collect::<Vec<T>>()
    }
}

impl<T: FloatCore> DenseMat<T> {
    pub fn new(nx: usize, ny: usize, data: Vec<T>) -> Self {
        assert!(data.len() == nx * ny);
        Self { nx, ny, data }
    }

    pub fn zeros(nx: usize, ny: usize) -> Self {
        Self::new(nx, ny, vec![T::zero(); nx * ny])
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    pub fn is_square(&self) -> bool {
        self.shape().0 == self.shape().1
    }

    pub fn gaussian_elimination_solve(mut self, b: &[T]) -> Vec<T> {
        assert!(self.is_square());
        assert!(self.ny == b.len());

        if self.nx == 0 {
            return vec![];
        }

        let mut b = b.to_vec();
        let mut sol = vec![T::zero(); self.nx];

        for ix in 0..(self.nx - 1) {
            for iy in (ix + 1)..self.ny {
                let coe = -self[(ix, iy)] / self[(ix, ix)];
                self[(ix, iy)] = T::zero();
                for jx in (ix + 1)..self.nx {
                    self[(jx, iy)] = self[(jx, iy)] + self[(jx, ix)] * coe;
                }
                b[iy] = b[iy] + b[ix] * coe;
            }
        }

        for iy in (0..self.ny).rev() {
            sol[iy] = b[iy];
            for ix in (iy + 1)..self.nx {
                sol[iy] = sol[iy] - self[(ix, iy)] * sol[ix];
            }
            sol[iy] = sol[iy] / self[(iy, iy)];
        }

        sol
    }
}

impl<T: FloatCore> Index<(usize, usize)> for DenseMat<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.ny + index.1]
    }
}

impl<T: FloatCore> IndexMut<(usize, usize)> for DenseMat<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.ny + index.1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::{assert_abs_diff_eq, AbsDiffEq};

    fn hilbert_mat<T: FloatCore>(n: usize) -> DenseMat<T> {
        let data = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| T::one() / T::from(i + j + 1).unwrap())
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .concat();
        DenseMat::new(n, n, data)
    }

    #[test]
    fn test_gaussian_elimination_solve_0() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
        let b = vec![3., 3., -6.];
        let sol = mat.gaussian_elimination_solve(&b);
        assert_abs_diff_eq!(sol[0], 3.0);
        assert_abs_diff_eq!(sol[1], 1.0);
        assert_abs_diff_eq!(sol[2], 2.0);
    }

    #[test]
    fn test_gaussian_elimination_solve_1() {
        let mat = DenseMat::new(1, 1, vec![2.]);
        let b = vec![3.];
        let sol = mat.gaussian_elimination_solve(&b);
        assert_abs_diff_eq!(sol[0], 1.5);
        assert!(sol.len() == 1);
    }

    #[test]
    fn test_gaussian_elimination_solve_2() {
        let mat = DenseMat::new(0, 0, Vec::<f64>::new());
        let b = vec![];
        let sol = mat.gaussian_elimination_solve(&b);
        assert!(sol.is_empty());
    }

    #[test]
    fn test_gaussian_elimination_solve_3() {
        let n = 6;
        let mat = hilbert_mat::<f64>(n);
        let x = vec![1.0; n];
        let b = mat.mul_vec(&x);
        let sol = mat.gaussian_elimination_solve(&b);
        sol.iter().for_each(|x| assert!(x.abs_diff_eq(&1.0, 1e-9)));

        let n = 8;
        let mat = hilbert_mat::<f64>(n);
        let x = vec![1.0; n];
        let b = mat.mul_vec(&x);
        let sol = mat.gaussian_elimination_solve(&b);
        sol.iter().for_each(|x| assert!(x.abs_diff_eq(&1.0, 1e-6)));
    }
}
