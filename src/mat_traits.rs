use std::iter::Sum;

pub trait MatShape {
    fn shape(&self) -> (usize, usize);

    fn nx(&self) -> usize {
        self.shape().0
    }

    fn ny(&self) -> usize {
        self.shape().1
    }

    fn is_square(&self) -> bool {
        self.shape().0 == self.shape().1
    }

    fn shape_eq(&self, other: &Self) -> bool {
        self.shape().0 == other.shape().0 && self.shape().1 == other.shape().1
    }
}

pub trait MatOps<T: crate::FloatCore + Sum>: MatShape {
    fn mul_vec(&self, b: &[T]) -> Vec<T>;

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

#[cfg(test)]
mod tests {
    use approx::AbsDiffEq;

    use super::*;

    use crate::dense_mat::DenseMat;
    use crate::sparse_mat::SparseMat;

    #[test]
    fn test_conjugate_gradient_iterate_0() {
        let mat = DenseMat::new(2, 2, vec![2., 2., 2., 5.]);
        let b = vec![6., 3.];
        let mut x = vec![0., 0.];
        let mut r = b
            .iter()
            .zip(mat.mul_vec(&x).into_iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let mut d = r.clone();

        mat.conjugate_gradient_iterate(&mut x, &mut r, &mut d);
        println!("iter 1, x = {x:?},  r = {r:?},  d = {d:?}");
        mat.conjugate_gradient_iterate(&mut x, &mut r, &mut d);
        println!("iter 2, x = {x:?},  r = {r:?},  d = {d:?}");

        x.iter()
            .zip([4., -1.].into_iter())
            .for_each(|(a, b)| assert!(a.abs_diff_eq(&b, 1e-14)))
    }

    #[test]
    fn test_conjugate_gradient_iterate_1() {
        let mat = SparseMat::full_from_vec(2, 2, vec![2., 2., 2., 5.]);
        let b = vec![6., 3.];
        let mut x = vec![0., 0.];
        let mut r = b
            .iter()
            .zip(mat.mul_vec(&x).into_iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let mut d = r.clone();

        mat.conjugate_gradient_iterate(&mut x, &mut r, &mut d);
        println!("iter 1, x = {x:?},  r = {r:?},  d = {d:?}");
        mat.conjugate_gradient_iterate(&mut x, &mut r, &mut d);
        println!("iter 2, x = {x:?},  r = {r:?},  d = {d:?}");

        x.iter()
            .zip([4., -1.].into_iter())
            .for_each(|(a, b)| assert!(a.abs_diff_eq(&b, 1e-14)))
    }

    #[test]
    fn test_conjugate_gradient_iterate_2() {
        let n = 100_000;

        let elems_0 = (0..n).map(|i| (i, n - i - 1, 0.5));
        let elems_1 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
        let elems_2 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));
        let elems_3 = (0..n).map(|i| (i, i, 3.0));

        let elems = elems_0.chain(elems_1).chain(elems_2).chain(elems_3);
        let mat = SparseMat::new(n, n, elems);

        let b = {
            let mut b = vec![1.5; n];
            b[0] = 2.5;
            b[n - 1] = 2.5;
            b[n / 2 - 1] = 1.0;
            b[n / 2] = 1.0;
            b
        };

        let mut x = vec![0.; n];
        let mut r = b
            .iter()
            .zip(mat.mul_vec(&x).into_iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let mut d = r.clone();

        let start = std::time::Instant::now();
        let iter_count = 20;
        for _ in 0..iter_count {
            mat.conjugate_gradient_iterate(&mut x, &mut r, &mut d);
        }
        let sol_norm = x
            .iter()
            .map(|a| f64::abs(a - 1.0))
            .reduce(f64::max)
            .unwrap();
        println!(
            "n = {n}, time used in {iter_count} conjugate gradient iteration = {} ms, solution infinity norm = {:e}",
            (std::time::Instant::now() - start).as_millis(),sol_norm
        );

        assert!(sol_norm < 1e-9);
    }

    #[test]
    fn test_preconditioned_conjugate_gradient_iterate_0() {
        let n = 100_000;

        let elems_0 = (0..n).map(|i| (i, n - i - 1, 0.5));
        let elems_1 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
        let elems_2 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));
        let elems_3 = (0..n).map(|i| (i, i, 3.0));

        let elems = elems_0.chain(elems_1).chain(elems_2).chain(elems_3);
        let mat = SparseMat::new(n, n, elems);

        let b = {
            let mut b = vec![1.5; n];
            b[0] = 2.5;
            b[n - 1] = 2.5;
            b[n / 2 - 1] = 1.0;
            b[n / 2] = 1.0;
            b
        };

        let mut x = vec![0.; n];
        let mut r = b
            .iter()
            .zip(mat.mul_vec(&x).into_iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();

        let perdictioner_inv = SparseMat::new(n, n, (0..n).map(|i| (i, i, 1.0 / 3.0)));
        let mut d = perdictioner_inv.mul_vec(&r);
        let mut z = d.clone();

        let start = std::time::Instant::now();
        let iter_count = 20;
        for _ in 0..iter_count {
            mat.preconditioned_conjugate_gradient_iterate(
                &|r| perdictioner_inv.mul_vec(r),
                &mut x,
                &mut r,
                &mut d,
                &mut z,
            );
        }
        let sol_norm = x
            .iter()
            .map(|a| f64::abs(a - 1.0))
            .reduce(f64::max)
            .unwrap();
        println!(
            "n = {n}, time used in {iter_count} preconditioned conjugate gradient iteration = {} ms, solution infinity norm = {:e}",
            (std::time::Instant::now() - start).as_millis(),sol_norm
        );

        assert!(sol_norm < 1e-9);
    }
}
