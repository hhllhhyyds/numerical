use std::iter::Sum;

use super::{dense_mat::DenseMat, mat_shape::MatShape, sparse_mat::SparseMat};

pub trait MatMulVec<T: crate::FloatCore + Sum> {
    fn mul_vec(&self, b: &[T]) -> Vec<T>;
}

pub trait MatIterMethods<T: crate::FloatCore + Sum>: MatMulVec<T> + MatShape {
    fn jacobi_iterate(x_k: &mut [T], b: &[T], diagonal: &[T], l_plus_u: &Self) {
        assert!(l_plus_u.is_square());
        let n = l_plus_u.nx();
        assert!(x_k.len() == n);
        assert!(b.len() == n);
        assert!(diagonal.len() == n);

        let x = l_plus_u.mul_vec(x_k);
        (0..n).for_each(|i| x_k[i] = (b[i] - x[i]) / diagonal[i]);
    }

    fn conjugate_gradient_iterate(&self, x: &mut [T], r: &mut [T], d: &mut [T]) {
        assert!(self.is_square());
        let n = self.nx();
        assert!(n == x.len());
        assert!(n == r.len());
        assert!(n == d.len());

        let a_mul_d = self.mul_vec(d);
        let r_dot = r.iter().map(|&r_i| r_i * r_i).sum::<T>();
        let alpha = r_dot / d.iter().zip(a_mul_d.iter()).map(|(&a, &b)| a * b).sum();

        x.iter_mut()
            .zip(d.iter())
            .for_each(|(x_i, &d_i)| *x_i = *x_i + alpha * d_i);

        r.iter_mut()
            .zip(a_mul_d.iter())
            .for_each(|(r_i, &a_mul_d_i)| *r_i = *r_i - alpha * a_mul_d_i);

        let beta = r.iter().map(|&r_i| r_i * r_i).sum::<T>() / r_dot;

        d.iter_mut()
            .zip(r.iter())
            .for_each(|(d_i, &r_i)| *d_i = r_i + beta * *d_i);
    }

    fn preconditioned_conjugate_gradient_iterate(
        &self,
        preconditioner_inv_mul: &dyn Fn(&[T]) -> Vec<T>,
        x: &mut [T],
        r: &mut [T],
        d: &mut [T],
        z: &mut [T],
    ) {
        assert!(self.is_square());
        let n = self.nx();
        assert!(n == x.len());
        assert!(n == r.len());
        assert!(n == d.len());
        assert!(n == z.len());

        let a_mul_d = self.mul_vec(d);
        let r_z_dot = r.iter().zip(z.iter()).map(|(&a, &b)| a * b).sum::<T>();
        let alpha = r_z_dot / d.iter().zip(a_mul_d.iter()).map(|(&a, &b)| a * b).sum();

        x.iter_mut()
            .zip(d.iter())
            .for_each(|(x_i, &d_i)| *x_i = *x_i + alpha * d_i);

        r.iter_mut()
            .zip(a_mul_d)
            .for_each(|(r_i, a_mul_d_i)| *r_i = *r_i - alpha * a_mul_d_i);

        let v = preconditioner_inv_mul(r);
        assert!(v.len() == n);
        z.iter_mut()
            .zip(v)
            .for_each(|(z_i, z_next_i)| *z_i = z_next_i);

        let beta = r.iter().zip(z.iter()).map(|(&a, &b)| a * b).sum::<T>() / r_z_dot;

        d.iter_mut()
            .zip(z.iter())
            .for_each(|(d_i, &z_i)| *d_i = z_i + beta * *d_i);
    }
}

impl<T: crate::FloatCore + Sum> MatIterMethods<T> for DenseMat<T> {}
impl<T: crate::FloatCore + Sum> MatIterMethods<T> for SparseMat<T> {}
