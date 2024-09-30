use approx::AbsDiffEq;

use numerical_algos::matrix::dense_mat::DenseMat;
use numerical_algos::matrix::mat_traits::{MatIterMethods, MatMulVec};
use numerical_algos::matrix::sparse_mat::SparseMat;

#[test]
fn test_conjugate_gradient_iterate_0() {
    let mat = DenseMat::from_slice(2, 2, &[2., 2., 2., 5.]);
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
