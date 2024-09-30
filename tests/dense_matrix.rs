use numerical_algos::matrix::dense_mat::DenseMat;
use numerical_algos::matrix::mat_traits::MatMulVec;
use numerical_algos::FloatCore;

use approx::assert_abs_diff_eq;
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

    pub fn hibert_mat<T: FloatCore>(n: usize) -> DenseMat<T> {
        let data = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| T::one() / T::from(i + j + 1).unwrap())
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .concat();
        DenseMat::from_vec(n, n, data)
    }
}

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
fn test_mul_mat() {
    let m0 = DenseMat::from_slice(1, 4, &[1.0; 4]);
    let m1 = DenseMat::from_slice(4, 1, &[1.0; 4]);

    let m3 = DenseMat::from_slice(1, 1, &[4.0; 1]);
    let m4 = DenseMat::from_slice(4, 4, &[1.0; 16]);

    assert!(m0.mul_mat(&m1).abs_diff_eq(&m4, 1e-14));
    assert!(m1.mul_mat(&m0).abs_diff_eq(&m3, 1e-14));
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

        assert!(helper::vec_abs_diff_eq(&x_0, &x_1, 100.0 * f64::EPSILON))
    })
}

#[test]
fn test_cholesky_factorization_0() {
    let mut mat = DenseMat::from_vec(3, 3, vec![4., -2., 2., -2., 2., -4., 2., -4., 11.]);
    let r = mat.cholesky_factorization().unwrap();
    let sol = DenseMat::from_vec(3, 3, vec![2., 0., 0., -1., 1., 0., 1., -3., 1.]);
    assert!(r.abs_diff_eq(&sol, 1e-14))
}

#[test]
fn test_cholesky_factorization_1() {
    (9..11).for_each(|n| {
        let mat = helper::random_symmetric_positive_definite_matrix::<f64>(n);
        let r_mat = mat.clone().cholesky_factorization().unwrap();
        let r_mat_inv = r_mat.transpose();

        assert!(r_mat_inv
            .mul_mat(&r_mat)
            .abs_diff_eq(&mat, 100. * f64::EPSILON))
    })
}

#[test]
fn test_gaussian_elimination_solve_0() {
    let mat = DenseMat::from_vec(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
    let b = vec![3., 3., -6.];
    let sol = mat.gaussian_elimination_solve(&b);
    assert_abs_diff_eq!(sol[0], 3.0);
    assert_abs_diff_eq!(sol[1], 1.0);
    assert_abs_diff_eq!(sol[2], 2.0);
}

#[test]
fn test_gaussian_elimination_solve_1() {
    let mat = DenseMat::from_vec(1, 1, vec![2.]);
    let b = vec![3.];
    let sol = mat.gaussian_elimination_solve(&b);
    assert_abs_diff_eq!(sol[0], 1.5);
    assert!(sol.len() == 1);
}

#[test]
fn test_gaussian_elimination_solve_2() {
    let mat = DenseMat::from_vec(0, 0, Vec::<f64>::new());
    let b = vec![];
    let sol = mat.gaussian_elimination_solve(&b);
    assert!(sol.is_empty());
}

#[test]
fn test_gaussian_elimination_solve_3() {
    let n = 6;
    let mat = helper::hibert_mat::<f64>(n);
    let x = vec![1.0; n];
    let b = mat.mul_vec(&x);
    let sol = mat.gaussian_elimination_solve(&b);
    sol.iter().for_each(|x| assert!(x.abs_diff_eq(&1.0, 1e-9)));

    let n = 8;
    let mat = helper::hibert_mat::<f64>(n);
    let x = vec![1.0; n];
    let b = mat.mul_vec(&x);
    let sol = mat.gaussian_elimination_solve(&b);
    sol.iter().for_each(|x| assert!(x.abs_diff_eq(&1.0, 1e-6)));
}

#[test]
fn test_gaussian_elimination_solve_4() {
    let mat = DenseMat::from_vec(3, 3, vec![1., 2., -3., -1., -2., 3., -1., -2., 1.]);
    let b = vec![3., 3., -6.];
    let sol = mat.gaussian_elimination_solve(&b);
    assert!(sol.iter().filter(|x| FloatCore::is_nan(**x)).count() > 0);
    println!("sol = {sol:?}")
}

#[test]
fn test_lu_0() {
    let mat = DenseMat::from_vec(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
    let (l_mat, u_mat) = mat.lu();
    assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
}

#[test]
fn test_lu_1() {
    let mat = DenseMat::from_vec(3, 3, vec![3., 6., 3., 1., 3., 1., 2., 4., 5.]);
    let (l_mat, u_mat) = mat.lu();
    assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
}

#[test]
fn test_lu_2() {
    let n = 8;
    let mat = helper::hibert_mat::<f64>(n);
    let (l_mat, u_mat) = mat.lu();
    assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
}

#[test]
fn test_lu_3() {
    let mat = DenseMat::from_vec(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
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
    let mat = helper::hibert_mat::<f64>(n);
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
    let mat = DenseMat::from_vec(3, 3, vec![1., 2., -3., -1., -2., 3., -1., -2., 1.]);
    let (l_mat, u_mat) = mat.lu();
    assert!(l_mat.contains_nan());
    assert!(u_mat.contains_nan());
}

#[test]
fn test_plu_0() {
    let mat = DenseMat::from_vec(3, 3, vec![2., 4., 1., 1., 4., 3., 5., -4., 1.]);
    let (p, l_mat, u_mat) = mat.plu();
    println!("p = {p:?}");
    println!("l_mat = {l_mat:?}");
    println!("u_mat = {u_mat:?}");
}

#[test]
fn test_plu_1() {
    let mat = DenseMat::from_vec(3, 3, vec![2., 4., 1., 1., 4., 3., 5., -4., 1.]);
    let b = [5., 0., 6.];
    let sol_0 = mat.lu_solve(&b);
    let sol_1 = mat.plu_solve(&b);

    sol_0
        .iter()
        .zip(sol_1.iter())
        .for_each(|(a, b)| assert!(a == b));
}

#[test]
fn test_plu_2() {
    let mat = DenseMat::from_vec(2, 2, vec![1e-20, 1., 1., 2.]);
    let b = [1., 4.];
    let sol_1 = mat.plu_solve(&b);
    let sol_0 = mat.gaussian_elimination_solve(&b);

    sol_0
        .iter()
        .zip(&[0., 1.])
        .for_each(|(a, b)| assert!(a.abs_diff_eq(b, 1e-10)));

    sol_1
        .iter()
        .zip(&[2., 1.])
        .for_each(|(a, b)| assert!(a.abs_diff_eq(b, 1e-10)));
}

#[test]
fn test_mat_inverse_0() {
    let mat = DenseMat::from_vec(3, 3, vec![2., 4., 1., 1., 4., 3., 5., -4., 1.]);
    let mat_inv = mat.inverse();
    let mat_mul_inv = mat.mul_mat(&mat_inv);
    assert!(mat_mul_inv.abs_diff_eq(&DenseMat::identity(3), 1e-14))
}

#[test]
fn test_mat_inverse_1() {
    let mat = DenseMat::from_vec(2, 2, vec![1., 1.0001, 1., 1.]);
    let cond = mat.condition_number().unwrap();
    assert!(cond.abs_diff_eq(&40004., 1e-3))
}

#[test]
fn test_mat_inverse_2() {
    let mat = helper::hibert_mat::<f64>(6);
    let cond = mat.condition_number().unwrap();
    println!("condition number of hilbert 6 = {cond:e}");
    assert!(cond.abs_diff_eq(&2.9070279e7, 1.0));

    let mat = helper::hibert_mat::<f64>(10);
    let cond = mat.condition_number().unwrap();
    println!("condition number of hilbert 10 = {cond:e}");
}

#[test]
fn test_strictly_diagonally_dominant() {
    let mat_0 = DenseMat::from_vec(3, 3, vec![3., 2., 1., 1., -5., 6., -1., 2., 8.]);
    assert!(mat_0.is_strictly_diagonally_dominant());

    let mut mat_1 = DenseMat::from_vec(3, 3, vec![3., 1., 9., 2., 8., 2., 6., 1., -2.]);
    assert!(!mat_1.is_strictly_diagonally_dominant());

    mat_1.swap_iy(0, 2);
    assert!(mat_1.is_strictly_diagonally_dominant());
}

#[test]
fn test_transpose() {
    let mat_0 = DenseMat::from_vec(2, 3, vec![3., 2., 1., 1., -5., 6.]);
    let mat_1 = DenseMat::from_vec(3, 2, vec![3., 1., 2., -5., 1., 6.]);
    let mat_0_transpose = mat_0.transpose();

    println!("mat_1 = {mat_1:?}, mat_0 transpose = {:?}", mat_0_transpose);
    assert!(mat_0_transpose.abs_diff_eq(&mat_1, f64::EPSILON))
}

#[test]
fn test_mat_add_sub() {
    let mat_0 = DenseMat::from_vec(2, 3, vec![3., 2., 1., 1., -5., 6.]);
    let mat_1 = DenseMat::from_vec(2, 3, vec![6., 4., 2., 2., -10., 12.]);
    let mat_2 = &mat_0 + &mat_0;
    let mat_3 = &mat_0 - &mat_0;
    assert!(mat_2.abs_diff_eq(&mat_1, f64::EPSILON));
    assert!(mat_3.abs_diff_eq(&DenseMat::zeros(2, 3), f64::EPSILON));
}

// TODO
// #[test]
// fn test_jacobi_iterate_0() {
//     let mat = DenseMat::from_vec(2, 2, vec![3., 1., 1., 2.]);
//     let diagonal = mat.diagonal();
//     let l_plus_u = &mat - &(DenseMat::from_diagonal(&diagonal));
//     let mut x = vec![0.0; 2];
//     let b = [5.0; 2];

//     (0..20).for_each(|_| {
//         DenseMat::jacobi_iterate(&mut x, &b, &diagonal, &l_plus_u);
//     });

//     (0..x.len()).for_each(|i| assert!(x[i].abs_diff_eq(&[1.0, 2.0][i], 1e-7)))
// }

#[test]
fn test_guass_seidel_iterate_0() {
    let mat = DenseMat::from_vec(3, 3, vec![3., 2., -1., 1., 4., 2., -1., 1., 5.]);

    let mut x = vec![0.0; 3];
    let b = [4., 1., 1.];

    (0..21).for_each(|i| {
        mat.gauss_seidel_iterate(&mut x, &b);
        if i < 2 {
            println!("x = {x:?}")
        }
    });

    (0..x.len()).for_each(|i| assert!(x[i].abs_diff_eq(&[2., -1., 1.][i], 1e-7)))
}

#[test]
fn test_nonlinear_newton_method_0() {
    let f = |x: &Vec<f64>| -> Vec<f64> {
        assert!(x.len() == 2);
        let (u, v) = (x[0], x[1]);
        vec![v - u.powi(3), u.powi(2) + v.powi(2) - 1.]
    };
    let df = |x: &Vec<f64>| -> DenseMat<f64> {
        assert!(x.len() == 2);
        let (u, v) = (x[0], x[1]);
        DenseMat::from_vec(2, 2, vec![-3. * u.powi(2), 2. * u, 1., 2. * v])
    };

    let mut x = vec![1., 2.];

    for _ in 0..6 {
        let s = df(&x).plu_solve(&f(&x).iter().map(|x| -x).collect::<Vec<_>>());
        x.iter_mut().zip(s.iter()).for_each(|(a, b)| *a += b);

        println!("x = {x:?}")
    }

    let y = f(&x);
    y.iter()
        .for_each(|val| assert!(val.abs_diff_eq(&0.0, 1e-14)))
}

#[test]
fn test_nonlinear_newton_method_1() {
    let f = |x: &Vec<f64>| -> Vec<f64> {
        assert!(x.len() == 2);
        let (u, v) = (x[0], x[1]);
        vec![
            6. * u.powi(3) + u * v - 3. * v.powi(3) - 4.,
            u.powi(2) - 18. * u * v.powi(2) + 16. * v.powi(3) + 1.,
        ]
    };
    let df = |x: &Vec<f64>| -> DenseMat<f64> {
        assert!(x.len() == 2);
        let (u, v) = (x[0], x[1]);
        DenseMat::from_vec(
            2,
            2,
            vec![
                18. * u.powi(2) + v,
                2. * u - 18. * v.powi(2),
                u - 9. * v.powi(2),
                -36. * u * v + 48. * v.powi(2),
            ],
        )
    };

    let mut x = vec![2., 2.];

    for _ in 0..6 {
        let s = df(&x).plu_solve(&f(&x).iter().map(|x| -x).collect::<Vec<_>>());
        x.iter_mut().zip(s.iter()).for_each(|(a, b)| *a += b);

        println!("x = {x:?}, y = {:?}", f(&x));
    }

    let y = f(&x);
    y.iter()
        .for_each(|val| assert!(val.abs_diff_eq(&0.0, 1e-14)))
}
