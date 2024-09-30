use numerical_algos::matrix::mat_traits::*;
use numerical_algos::matrix::sparse_mat::SparseMat;

use plotters::prelude::*;

fn main() {
    let n = 500;

    let elems_d = (0..n).map(|i| (i, i, ((i + 1) as f64).sqrt()));
    let elems_l = (0..(n - 10)).map(|i| (i, i + 10, (i as f64).cos()));
    let elems_u = (0..(n - 10)).map(|i| (i + 10, i, (i as f64).cos()));
    let mat = SparseMat::new(
        n,
        n,
        elems_d
            .clone()
            .chain(elems_u.clone())
            .chain(elems_l.clone()),
    );

    let preconditioner_identity_inv = SparseMat::new(n, n, (0..n).map(|i| (i, i, 1.0)));
    let preconditioner_jacobi_inv =
        SparseMat::new(n, n, (0..n).map(|i| (i, i, 1.0 / ((i + 1) as f64).sqrt())));

    let omega = 1.2;
    let l_mat = &SparseMat::new(n, n, elems_l).mul_diagonal(
        &elems_d
            .clone()
            .map(|(_, _, val)| omega / val)
            .collect::<Vec<_>>(),
    ) + &SparseMat::new(n, n, (0..n).map(|i| (i, i, 1.0)));
    let u_mat = &SparseMat::new(n, n, elems_d)
        + &SparseMat::new(n, n, elems_u.map(|(ix, iy, val)| (ix, iy, omega * val)));
    let preconditioner_gauss_seidel_inv_mul_fn = |v: &[f64]| {
        assert!(v.len() == n);
        let c = l_mat.back_substitute_lower_triangle(v);
        u_mat.back_substitute_upper_triangle(&c)
    };

    let b = mat.mul_vec(&vec![1.0; n]);

    let mut x_a = vec![0.0; n];
    let mut r_a = b
        .iter()
        .zip(mat.mul_vec(&x_a).into_iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let mut d_a = preconditioner_identity_inv.mul_vec(&r_a);
    let mut z_a = d_a.clone();

    let mut x_b = x_a.clone();
    let mut r_b = r_a.clone();
    let mut d_b = preconditioner_jacobi_inv.mul_vec(&r_b);
    let mut z_b = d_b.clone();

    let mut x_c = x_a.clone();
    let mut r_c = r_a.clone();
    let mut d_c = preconditioner_gauss_seidel_inv_mul_fn(&r_c);
    let mut z_c = d_c.clone();

    let iter_count = 40;
    let mut diff_a_arr = vec![];
    let mut diff_b_arr = vec![];
    let mut diff_c_arr = vec![];

    let start = std::time::Instant::now();
    for _ in 0..iter_count {
        mat.preconditioned_conjugate_gradient_iterate(
            &|r| preconditioner_identity_inv.mul_vec(r),
            &mut x_a,
            &mut r_a,
            &mut d_a,
            &mut z_a,
        );
        diff_a_arr.push(
            x_a.iter()
                .map(|val| (val - 1.0).abs())
                .reduce(f64::max)
                .unwrap(),
        );
    }
    println!(
        "n = {n}, time used in {iter_count} conjugate gradient iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    let start = std::time::Instant::now();
    for _ in 0..iter_count {
        mat.preconditioned_conjugate_gradient_iterate(
            &|r| preconditioner_jacobi_inv.mul_vec(r),
            &mut x_b,
            &mut r_b,
            &mut d_b,
            &mut z_b,
        );
        diff_b_arr.push(
            x_b.iter()
                .map(|val| (val - 1.0).abs())
                .reduce(f64::max)
                .unwrap(),
        );
    }
    println!(
        "n = {n}, time used in {iter_count} jacobi precoditioner conjugate gradient iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    let start = std::time::Instant::now();
    for _ in 0..iter_count {
        mat.preconditioned_conjugate_gradient_iterate(
            &preconditioner_gauss_seidel_inv_mul_fn,
            &mut x_c,
            &mut r_c,
            &mut d_c,
            &mut z_c,
        );
        diff_c_arr.push(
            x_c.iter()
                .map(|val| (val - 1.0).abs())
                .reduce(f64::max)
                .unwrap(),
        );
    }
    println!(
        "n = {n}, time used in {iter_count} gauss seidel precoditioner conjugate gradient iteration = {} ms",
        (std::time::Instant::now() - start).as_millis()
    );

    // 创建svg图片对象
    let root = SVGBackend::new("plot.svg", (640, 480)).into_drawing_area();
    // 图片对象的背景颜色填充
    root.fill(&WHITE).unwrap();
    // 创建绘图对象
    let mut chart = ChartBuilder::on(&root)
        // 图表名称  (字体样式, 字体大小)
        .caption(
            "Conjugate gradient method preconditioner comparition",
            ("sans-serif", 30),
        )
        // 图表左侧与图片边缘的间距
        .set_label_area_size(LabelAreaPosition::Left, 40)
        // 图表底部与图片边缘的间距
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        // 构建二维图像, x轴 0.0 - 10.0； y轴 0.0 - 10.0；
        .build_cartesian_2d(0.0..40.0, -16.0..1.0)
        .unwrap();
    // 配置网格线
    chart.configure_mesh().draw().unwrap();

    // 设置三角形标记散点
    chart
        .draw_series(
            diff_a_arr
                .iter()
                .enumerate()
                .map(|point| Circle::new((point.0 as f64, point.1.log10()), 5, &BLUE)),
        )
        .unwrap()
        .label("non preconditioner")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    // 设置圆圈标记散点
    chart
        .draw_series(
            diff_b_arr
                .iter()
                .enumerate()
                .map(|point| TriangleMarker::new((point.0 as f64, point.1.log10()), 5, &RED)),
        )
        .unwrap()
        .label("jacobi perconditioner")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    // 设置圆圈标记散点
    chart
        .draw_series(
            diff_c_arr
                .iter()
                .enumerate()
                .map(|point| Circle::new((point.0 as f64, point.1.log10()), 5, &GREEN)),
        )
        .unwrap()
        .label("gauss seidel perconditioner")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // 配置标签样式
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}
