use numerical_algos::polynomial::*;

use plotters::prelude::*;

fn main() {
    let n = 15;

    let interpolate_points = (0..n)
        .map(|i| {
            let space = 2.0 / (n - 1) as f64;
            let x = -1.0 + i as f64 * space;
            let y = 1.0 / (1.0 + 12.0 * x * x);
            (x, y)
        })
        .collect::<Vec<_>>();

    let interpolation_poly =
        PolynomialInterpolator::interpolation_series(interpolate_points.clone().into_iter(), 1e-12);

    let poly_plot_points = (0..(n * 30))
        .map(|i| {
            let space = 2.4 / (n * 30) as f64;
            let x = -1.2 + i as f64 * space;
            let y = interpolation_poly.poly().nest_multiply(x);
            (x, y)
        })
        .collect::<Vec<_>>();

    let pic = SVGBackend::new("plot.svg", (6400, 4800)).into_drawing_area();
    pic.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&pic)
        .caption("Runge phenomenon", ("sans-serif", 300))
        .set_all_label_area_size(400)
        .build_cartesian_2d(-1.1..1.1, -3.0..3.0)
        .unwrap();
    chart
        .configure_mesh()
        .bold_line_style(RGBColor(100, 100, 100).filled().stroke_width(10))
        .light_line_style(RGBColor(100, 100, 100).filled().stroke_width(3))
        .label_style(("sans-serif", 200))
        .draw()
        .unwrap();

    chart
        .draw_series(
            interpolate_points
                .iter()
                .map(|point| Circle::new((point.0, point.1), 30, RED.filled())),
        )
        .unwrap()
        .label("interpolation points");

    chart
        .draw_series(LineSeries::new(
            poly_plot_points,
            BLUE.filled().stroke_width(16),
        ))
        .unwrap()
        .label("interpolation polynomial");
}
