use std::{
    iter::Sum,
    ops::{Index, IndexMut},
};

use approx::AbsDiffEq;

use crate::FloatCore;

#[derive(Debug, Clone, PartialEq)]
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

    pub fn mul_mat(&self, other: &Self) -> Self {
        assert!(self.nx == other.ny);

        let nx = other.nx;
        let ny = self.ny;
        let data = (0..nx)
            .map(|ix| self.mul_vec(&other.data[(ix * other.ny)..((ix + 1) * other.ny)]))
            .collect::<Vec<_>>()
            .concat();

        Self::new(nx, ny, data)
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

        for ix in 0..self.nx {
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

    pub fn lu(&self) -> (Self, Self) {
        assert!(self.is_square());

        let mut u_mat = self.clone();
        let mut l_mat = Self::new(self.nx, self.ny, vec![T::zero(); self.nx * self.ny]);

        for ix in 0..self.nx {
            l_mat[(ix, ix)] = T::one();
            for iy in (ix + 1)..self.ny {
                let coe = u_mat[(ix, iy)] / u_mat[(ix, ix)];
                u_mat[(ix, iy)] = T::zero();
                l_mat[(ix, iy)] = coe;
                for jx in (ix + 1)..self.nx {
                    u_mat[(jx, iy)] = u_mat[(jx, iy)] - u_mat[(jx, ix)] * coe;
                }
            }
        }

        (l_mat, u_mat)
    }

    pub fn back_substitute_lower_triangle(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());
        assert!(self.is_square());

        let mut y = vec![];
        for iy in 0..self.ny {
            let mut val = b[iy];
            for ix in 0..iy {
                val = val - self[(ix, iy)] * y[ix];
            }
            val = val / self[(iy, iy)];
            y.push(val);
        }

        y
    }

    pub fn back_substitute_upper_triangle(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());
        assert!(self.is_square());

        let mut y = vec![T::zero(); self.ny];

        for iy in (0..self.ny).rev() {
            y[iy] = b[iy];
            for ix in (iy + 1)..self.nx {
                y[iy] = y[iy] - self[(ix, iy)] * y[ix];
            }
            y[iy] = y[iy] / self[(iy, iy)];
        }

        y
    }

    pub fn lu_solve(&self, b: &[T]) -> Vec<T> {
        let (l, u) = self.lu();
        let c = l.back_substitute_lower_triangle(b);
        u.back_substitute_upper_triangle(&c)
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

impl<T: FloatCore + AbsDiffEq<Epsilon = T>> AbsDiffEq for DenseMat<T> {
    type Epsilon = T;

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.nx == other.nx
            && self.ny == other.nx
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

    #[test]
    fn test_gaussian_elimination_solve_4() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., -1., -2., 3., -1., -2., 1.]);
        let b = vec![3., 3., -6.];
        let sol = mat.gaussian_elimination_solve(&b);
        assert!(sol.iter().filter(|x| x.is_nan()).count() > 0);
        println!("sol = {sol:?}")
    }

    #[test]
    fn test_lu_0() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
        let (l_mat, u_mat) = mat.lu();
        assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
    }

    #[test]
    fn test_lu_1() {
        let mat = DenseMat::new(3, 3, vec![3., 6., 3., 1., 3., 1., 2., 4., 5.]);
        let (l_mat, u_mat) = mat.lu();
        assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
    }

    #[test]
    fn test_lu_2() {
        let n = 8;
        let mat = hilbert_mat::<f64>(n);
        let (l_mat, u_mat) = mat.lu();
        assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
    }

    #[test]
    fn test_lu_3() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
        let b = vec![3., 3., -6.];
        let sol_0 = mat.lu_solve(&b);
        let sol_1 = mat.gaussian_elimination_solve(&b);
        sol_0
            .iter()
            .zip(sol_1.iter())
            .for_each(|(a, b)| assert!(a == b));
    }

    #[test]
    fn test_lu_4() {
        let n = 8;
        let mat = hilbert_mat::<f64>(n);
        let b = vec![1.0; n];
        let sol_0 = mat.lu_solve(&b);
        let sol_1 = mat.gaussian_elimination_solve(&b);
        sol_0
            .iter()
            .zip(sol_1.iter())
            .for_each(|(a, b)| assert!(a == b));
        println!("sol = {sol_0:?}")
    }

    #[test]
    fn test_lu_5() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., -1., -2., 3., -1., -2., 1.]);
        let (l_mat, u_mat) = mat.lu();
        assert!(l_mat.data.iter().filter(|x| x.is_nan()).count() > 0);
        assert!(u_mat.data.iter().filter(|x| x.is_nan()).count() > 0);
    }
}
