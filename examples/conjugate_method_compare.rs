use numerical_algos::mat_traits::*;
use numerical_algos::sparse_mat::SparseMat;

fn main() {
    let n = 500;

    let elems_0 = (0..n).map(|i| (i, i, ((i + 1) as f64).sqrt()));
    let elems_1 = (0..(n - 10)).map(|i| (i, i + 10, (i as f64).cos()));
    let elems_2 = (0..(n - 10)).map(|i| (i + 10, i, (i as f64).cos()));
    let mat = SparseMat::new(n, n, elems_0.chain(elems_1).chain(elems_2));

    let preconditioner_identity_inv = SparseMat::new(n, n, (0..n).map(|i| (i, i, 1.0)));
    let preconditioner_jacobi_inv =
        SparseMat::new(n, n, (0..n).map(|i| (i, i, 1.0 / ((i + 1) as f64).sqrt())));

    let b = mat.mul_vec(&vec![1.0; n]);

    let mut x_a = vec![0.0; n];
    let mut r_a = b
        .iter()
        .zip(mat.mul_vec(&x_a).into_iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let mut d_a = preconditioner_identity_inv.mul_vec(&r_a);
    let mut z_a = d_a.clone();

    let mut x_b = vec![0.0; n];
    let mut r_b = b
        .iter()
        .zip(mat.mul_vec(&x_b).into_iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let mut d_b = preconditioner_identity_inv.mul_vec(&r_b);
    let mut z_b = d_b.clone();

    for i in 0..40 {
        mat.preconditioned_conjugate_gradient_iterate(
            &preconditioner_identity_inv,
            &mut x_a,
            &mut r_a,
            &mut d_a,
            &mut z_a,
        );
        let diff_a = x_a
            .iter()
            .map(|val| (val - 1.0).abs())
            .reduce(f64::max)
            .unwrap();

        mat.preconditioned_conjugate_gradient_iterate(
            &preconditioner_jacobi_inv,
            &mut x_b,
            &mut r_b,
            &mut d_b,
            &mut z_b,
        );
        let diff_b = x_b
            .iter()
            .map(|val| (val - 1.0).abs())
            .reduce(f64::max)
            .unwrap();

        println!("iter {i}, diff a = {diff_a:e}, diff b = {diff_b:e}");
    }
}
