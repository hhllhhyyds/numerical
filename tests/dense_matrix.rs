use numerical_algos::dense_mat::DenseMat;
use numerical_algos::mat_traits::MatOps;
use numerical_algos::FloatCore;

use approx::AbsDiffEq;
use rand::Rng;
use std::iter::Sum;

mod helper {
    use super::*;

    pub fn vec_abs_diff_eq<T: FloatCore>(a: &[T], b: &[T], eps: T) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(a, b)| (*a - *b).abs() < eps)
    }

    pub fn random_inversable_square_matrix<T>(n: usize) -> DenseMat<T>
    where
        T: FloatCore + Sum,
    {
        let mut rng = rand::thread_rng();
        let mut mat = DenseMat::zeros(n, n);

        (0..n).for_each(|ix| {
            (0..n).for_each(|iy| mat[(ix, iy)] = T::from(rng.gen_range(-1.0..1.0)).unwrap())
        });

        let mut x = 1.0;
        while !mat.is_strictly_diagonally_dominant() {
            mat = &mat + &DenseMat::from_diagonal(&vec![T::from(x).unwrap(); n]);
            x *= 2.0;
        }

        mat
    }

    pub fn random_symmetric_positive_definite_matrix<T>(n: usize) -> DenseMat<T>
    where
        T: FloatCore + Sum,
    {
        let mut rng = rand::thread_rng();
        let mut mat = DenseMat::zeros(n, n);

        (0..n).for_each(|ix| {
            (0..ix).for_each(|iy| {
                mat[(ix, iy)] = T::from(rng.gen_range(-1.0..1.0)).unwrap();
                mat[(iy, ix)] = mat[(ix, iy)];
            })
        });

        let mut x = 1.0;
        while !mat.is_strictly_diagonally_dominant() {
            mat = &mat + &DenseMat::from_diagonal(&vec![T::from(x).unwrap(); n]);
            x *= 2.0;
        }

        mat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        (9..11).for_each(|n| {
            let mat = helper::random_inversable_square_matrix::<f64>(n);
            let b = vec![1.0; n];

            let x = mat.plu_solve(&b);
            let y = mat.mul_vec(&x);
            assert!(helper::vec_abs_diff_eq(&b, &y, 5.0 * f64::EPSILON));

            let x_1 = mat.clone().gaussian_elimination_solve(&b);
            assert!(helper::vec_abs_diff_eq(&x, &x_1, f64::EPSILON));

            let x_2 = mat.lu_solve(&b);
            assert!(helper::vec_abs_diff_eq(&x, &x_2, f64::EPSILON))
        })
    }

    #[test]
    fn test_mul_mat_inverse() {
        (9..11).for_each(|n| {
            let mat = helper::random_inversable_square_matrix::<f64>(n);
            let inv_mat = mat.inverse();

            assert!(mat
                .mul_mat(&inv_mat)
                .abs_diff_eq(&DenseMat::identity(n), 2.0 * f64::EPSILON))
        })
    }

    #[test]
    fn test_gauss_seidel_iterate() {
        (9..11).for_each(|n| {
            let mat = helper::random_inversable_square_matrix::<f64>(n);
            let b = vec![1.0; n];

            let x_0 = mat.plu_solve(&b);

            let mut x_1 = vec![1.0; n];
            (0..18).for_each(|_| mat.gauss_seidel_iterate(&mut x_1, &b));

            assert!(helper::vec_abs_diff_eq(&x_0, &x_1, f64::EPSILON))
        })
    }

    #[test]
    fn test_cholesky_factorization() {
        (9..11).for_each(|n| {
            let mat = helper::random_symmetric_positive_definite_matrix::<f64>(n);
            let r_mat = mat.clone().cholesky_factorization().unwrap();
            let r_mat_inv = r_mat.transpose();

            assert!(r_mat_inv
                .mul_mat(&r_mat)
                .abs_diff_eq(&mat, 10. * f64::EPSILON))
        })
    }
}
