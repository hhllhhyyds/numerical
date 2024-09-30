use approx::AbsDiffEq;

use numerical_algos::matrix::mat_traits::MatIterMethods;
use numerical_algos::matrix::{dense_mat::DenseMat, mat_traits::MatMulVec, sparse_mat::SparseMat};
use numerical_algos::FloatCore;

#[test]
#[ignore = "slow, run by hand"]
fn test_jacobi_iterate_0() {
    let n = 100_000;
    let elems_0 = (0..n).map(|i| (i, n - i - 1, 0.5));
    let elems_1 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
    let elems_2 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));

    let elems = elems_0.chain(elems_1).chain(elems_2);
    let mat = SparseMat::new(n, n, elems);

    let b = {
        let mut b = vec![1.5; n];
        b[0] = 2.5;
        b[n - 1] = 2.5;
        b[n / 2 - 1] = 1.0;
        b[n / 2] = 1.0;
        b
    };

    let mut x = vec![0.0; n];

    let diag = vec![3.0; n];

    let start = std::time::Instant::now();
    for _ in 0..50 {
        SparseMat::jacobi_iterate(&mut x, &b, &diag, &mat);
    }
    println!(
        "n = {n}, time used in 50 jacobi iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    assert!(x.iter().all(|y| y.abs_diff_eq(&1.0, 1e-6)));
}

#[test]
fn test_jacobi_iterate_1() {
    let mat = SparseMat::new(2, 2, [(0_usize, 1_usize, 1.0_f64), (1, 0, 1.0)].into_iter());
    let diagonal = vec![3.0, 2.0];
    let mut x = vec![0.0; 2];
    let b = [5.0; 2];

    (0..20).for_each(|_| {
        SparseMat::jacobi_iterate(&mut x, &b, &diagonal, &mat);
    });

    (0..x.len()).for_each(|i| assert!(x[i].abs_diff_eq(&[1.0, 2.0][i], 1e-7)))
}

#[test]
fn test_jacobi_iterate_2() {
    let n = 100;
    let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
    let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));

    let elems = elems_0.chain(elems_1);
    let mat = SparseMat::new(n, n, elems.clone());

    let b = {
        let mut b = vec![1.; n];
        b[0] = 2.;
        b[n - 1] = 2.;
        b
    };

    let mut x = vec![0.0; n];

    let diag = vec![3.0; n];

    let start = std::time::Instant::now();

    let mut iter_count = 0;
    while x
        .iter()
        .map(|val| (val - 1.).abs())
        .reduce(f64::max)
        .unwrap()
        >= 1e-6
    {
        iter_count += 1;
        SparseMat::jacobi_iterate(&mut x, &b, &diag, &mat);
    }

    println!(
        "n = {n}, time used in {iter_count} jacobi iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    let full_mat = SparseMat::new(n, n, elems.chain((0..n).map(|i| (i, i, 3.0))));
    let y = full_mat.mul_vec(&x);
    println!(
        "backward error = {:e}",
        y.iter()
            .zip(b.iter())
            .map(|(m, n)| (m - n).abs())
            .reduce(f64::max)
            .unwrap()
    );

    assert!(x.iter().all(|y| y.abs_diff_eq(&1.0, 1e-6)));
}

#[test]
fn test_jacobi_iterate_3() {
    let n = 100;
    let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, 1.0));
    let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, 1.0));

    let elems = elems_0.chain(elems_1);
    let mat = SparseMat::new(n, n, elems.clone());

    let b = {
        let mut b = vec![0.; n];
        b[0] = 1.;
        b[n - 1] = -1.;
        b
    };

    let mut x = vec![0.0; n];

    let diag = vec![2.0; n];

    let sol = (0..n)
        .map(|i| if i % 2 == 0 { 1. } else { -1. })
        .collect::<Vec<f64>>();

    let start = std::time::Instant::now();

    let mut iter_count = 0;
    while x
        .iter()
        .zip(sol.iter())
        .map(|(a, b)| (a - b).abs())
        .reduce(f64::max)
        .unwrap()
        >= 1e-3
    {
        iter_count += 1;
        SparseMat::jacobi_iterate(&mut x, &b, &diag, &mat);
    }

    println!(
        "n = {n}, time used in {iter_count} jacobi iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    let full_mat = SparseMat::new(n, n, elems.chain((0..n).map(|i| (i, i, 2.0))));
    let y = full_mat.mul_vec(&x);
    println!(
        "backward error = {:e}",
        y.iter()
            .zip(b.iter())
            .map(|(m, n)| (m - n).abs())
            .reduce(f64::max)
            .unwrap()
    );

    assert!(x
        .iter()
        .zip(sol.iter())
        .all(|(a, b)| a.abs_diff_eq(b, 1e-3)));
}

#[test]
fn test_guass_seidel_iterate_0() {
    let mat = SparseMat::full_from_vec(3, 3, vec![3., 2., -1., 1., 4., 2., -1., 1., 5.]);

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
fn test_guass_seidel_iterate_1() {
    let n = 100;
    let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
    let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));
    let elems_2 = (0..n).map(|i| (i, i, 3.0));

    let elems = elems_0.chain(elems_1).chain(elems_2);
    let mat = SparseMat::new(n, n, elems);

    let b = {
        let mut b = vec![1.; n];
        b[0] = 2.;
        b[n - 1] = 2.;
        b
    };

    let mut x = vec![0.0; n];

    let start = std::time::Instant::now();

    let mut iter_count = 0;
    while x
        .iter()
        .map(|val| (val - 1.).abs())
        .reduce(f64::max)
        .unwrap()
        >= 1e-6
    {
        iter_count += 1;
        mat.gauss_seidel_iterate(&mut x, &b);
    }

    println!(
        "n = {n}, time used in {iter_count} gauss seidel iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    let y = mat.mul_vec(&x);
    println!(
        "backward error = {:e}",
        y.iter()
            .zip(b.iter())
            .map(|(m, n)| (m - n).abs())
            .reduce(f64::max)
            .unwrap()
    );

    assert!(x.iter().all(|y| y.abs_diff_eq(&1.0, 1e-6)));
}

#[test]
fn test_guass_seidel_iterate_2() {
    let n = 100;
    let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, 1.0));
    let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, 1.0));
    let elems_2 = (0..n).map(|i| (i, i, 2.0));

    let elems = elems_0.chain(elems_1).chain(elems_2);
    let mat = SparseMat::new(n, n, elems.clone());

    let b = {
        let mut b = vec![0.; n];
        b[0] = 1.;
        b[n - 1] = -1.;
        b
    };

    let mut x = vec![0.0; n];

    let sol = (0..n)
        .map(|i| if i % 2 == 0 { 1. } else { -1. })
        .collect::<Vec<f64>>();

    let start = std::time::Instant::now();

    let mut iter_count = 0;
    while x
        .iter()
        .zip(sol.iter())
        .map(|(a, b)| (a - b).abs())
        .reduce(f64::max)
        .unwrap()
        >= 1e-3
    {
        iter_count += 1;
        mat.gauss_seidel_iterate(&mut x, &b);
    }

    println!(
        "n = {n}, time used in {iter_count} gauss seidel iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    let y = mat.mul_vec(&x);
    println!(
        "backward error = {:e}",
        y.iter()
            .zip(b.iter())
            .map(|(m, n)| (m - n).abs())
            .reduce(f64::max)
            .unwrap()
    );

    assert!(x
        .iter()
        .zip(sol.iter())
        .all(|(a, b)| a.abs_diff_eq(b, 1e-3)));
}

#[test]
fn test_sor_guass_seidel_iterate_0() {
    let n = 100;
    let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
    let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));
    let elems_2 = (0..n).map(|i| (i, i, 3.0));

    let elems = elems_0.chain(elems_1).chain(elems_2);
    let mat = SparseMat::new(n, n, elems);

    let b = {
        let mut b = vec![1.; n];
        b[0] = 2.;
        b[n - 1] = 2.;
        b
    };

    let mut x = vec![0.0; n];

    let start = std::time::Instant::now();

    let mut iter_count = 0;
    while x
        .iter()
        .map(|val| (val - 1.).abs())
        .reduce(f64::max)
        .unwrap()
        >= 1e-6
    {
        iter_count += 1;
        mat.sor_gauss_seidel_iterate(&mut x, &b, 1.2);
    }

    println!(
        "n = {n}, time used in {iter_count} sor gauss seidel iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    let y = mat.mul_vec(&x);
    println!(
        "backward error = {:e}",
        y.iter()
            .zip(b.iter())
            .map(|(m, n)| (m - n).abs())
            .reduce(f64::max)
            .unwrap()
    );

    assert!(x.iter().all(|y| y.abs_diff_eq(&1.0, 1e-6)));
}

#[test]
fn test_sor_guass_seidel_iterate_1() {
    let n = 100;
    let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, 1.0));
    let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, 1.0));
    let elems_2 = (0..n).map(|i| (i, i, 2.0));

    let elems = elems_0.chain(elems_1).chain(elems_2);
    let mat = SparseMat::new(n, n, elems.clone());

    let b = {
        let mut b = vec![0.; n];
        b[0] = 1.;
        b[n - 1] = -1.;
        b
    };

    let mut x = vec![0.0; n];

    let sol = (0..n)
        .map(|i| if i % 2 == 0 { 1. } else { -1. })
        .collect::<Vec<f64>>();

    let start = std::time::Instant::now();

    let mut iter_count = 0;
    while x
        .iter()
        .zip(sol.iter())
        .map(|(a, b)| (a - b).abs())
        .reduce(f64::max)
        .unwrap()
        >= 1e-3
    {
        iter_count += 1;
        mat.sor_gauss_seidel_iterate(&mut x, &b, 1.5);
    }

    println!(
        "n = {n}, time used in {iter_count} sor gauss seidel iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    let y = mat.mul_vec(&x);
    println!(
        "backward error = {:e}",
        y.iter()
            .zip(b.iter())
            .map(|(m, n)| (m - n).abs())
            .reduce(f64::max)
            .unwrap()
    );

    assert!(x
        .iter()
        .zip(sol.iter())
        .all(|(a, b)| a.abs_diff_eq(b, 1e-3)));
}

#[test]
fn test_back_substitute_lower_triangle_0() {
    let dense_mat = DenseMat::from_slice(3, 3, &[1., 2., 3., 0., 4., 5., 0., 0., 6.]);
    let sparse_mat = SparseMat::new(
        3,
        3,
        vec![
            (0, 0, 1.),
            (0, 1, 2.),
            (0, 2, 3.),
            (1, 1, 4.),
            (1, 2, 5.),
            (2, 2, 6.),
        ]
        .into_iter(),
    );

    let b = [1.3, -2.4, 3.5];

    let x_0 = dense_mat.back_substitute_lower_triangle(&b);
    let x_1 = sparse_mat.back_substitute_lower_triangle(&b);

    x_0.iter()
        .zip(x_1.iter())
        .for_each(|(a, b)| assert!(a == b))
}

#[test]
fn test_back_substitute_lower_triangle_1() {
    let dense_mat = DenseMat::from_slice(3, 3, &[1., 0., 3., 0., 4., 5., 0., 0., 6.]);
    let sparse_mat = SparseMat::new(
        3,
        3,
        vec![(0, 0, 1.), (0, 2, 3.), (1, 1, 4.), (1, 2, 5.), (2, 2, 6.)].into_iter(),
    );

    let b = [1.3, -2.4, 3.5];

    let x_0 = dense_mat.back_substitute_lower_triangle(&b);
    let x_1 = sparse_mat.back_substitute_lower_triangle(&b);

    x_0.iter()
        .zip(x_1.iter())
        .for_each(|(a, b)| assert!(a == b))
}

#[test]
fn test_back_substitute_upper_triangle_0() {
    let dense_mat = DenseMat::from_slice(3, 3, &[1., 0., 0., 2., 3., 0., 4., 5., 6.]);
    let sparse_mat = SparseMat::new(
        3,
        3,
        vec![
            (0, 0, 1.),
            (1, 0, 2.),
            (1, 1, 3.),
            (2, 0, 4.),
            (2, 1, 5.),
            (2, 2, 6.),
        ]
        .into_iter(),
    );

    let b = [1.3, -2.4, 3.5];

    let x_0 = dense_mat.back_substitute_upper_triangle(&b);
    let x_1 = sparse_mat.back_substitute_upper_triangle(&b);

    println!("x_0 = {x_0:?}, x_1 = {x_1:?}");

    x_0.iter()
        .zip(x_1.iter())
        .for_each(|(a, b)| assert!(a.abs_diff_eq(b, 1e-14)))
}

#[test]
fn test_back_substitute_upper_triangle_1() {
    let dense_mat = DenseMat::from_slice(3, 3, &[1., 0., 0., 2., 3., 0., 4., 0., 6.]);
    let sparse_mat = SparseMat::new(
        3,
        3,
        vec![(0, 0, 1.), (1, 0, 2.), (1, 1, 3.), (2, 0, 4.), (2, 2, 6.)].into_iter(),
    );

    let b = [1.3, -2.4, 3.5];

    let x_0 = dense_mat.back_substitute_upper_triangle(&b);
    let x_1 = sparse_mat.back_substitute_upper_triangle(&b);

    println!("x_0 = {x_0:?}, x_1 = {x_1:?}");

    x_0.iter()
        .zip(x_1.iter())
        .for_each(|(a, b)| assert!(a == b))
}
