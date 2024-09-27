use numerical_algos::polynomial::PolynomialInterpolator;

use approx::AbsDiffEq;
use rand::Rng;

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    #[test]
    fn test_polynomial_interpolator() {
        for n in 1..20 {
            let mut rng = rand::thread_rng();

            let data = (0..n)
                .map(|x| {
                    let region = 1.0 / (n as f64);
                    (
                        rng.gen_range(((x as f64) * region)..((x as f64 + 0.5) * region)),
                        rng.gen_range(-1.0..1.0),
                    )
                })
                .collect::<Vec<_>>();

            let mut interpolator = PolynomialInterpolator::new(data[0].0, data[0].1, 1e-12);
            for p in &data[1..data.len()] {
                interpolator.add_point(p.0, p.1);
            }

            for i in 0..data.len() {
                assert!(data[i]
                    .1
                    .abs_diff_eq(&interpolator.poly().nest_multiply(data[i].0), 1e-8))
            }
        }
    }
}
