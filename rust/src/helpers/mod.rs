use ndarray::Array2;

fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    let a = (x - mu) / sigma;
    (-0.5 * a * a).exp()
}

pub fn gaussian_kernel_2d(radius: u16, sigma_in: Option<f64>) -> Array2<f64> {
    let sigma = sigma_in.unwrap_or(radius as f64 / 2.);
    let size = 2 * radius as usize + 1;
    let mut kernel = Array2::zeros((size, size));
    let mut sum = 0.;
    for x in 0..size {
        for y in 0..size {
            let val =
                gaussian(x as f64, radius as f64, sigma) * gaussian(y as f64, radius as f64, sigma);
            kernel[[x, y]] = val;
            sum += val;
        }
    }
    for x in 0..size {
        for y in 0..size {
            kernel[[x, y]] /= sum;
        }
    }
    kernel
}

pub fn get_parabola_vertex(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> (f64, f64) {
    let denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
    let a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
    let b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom;
    let c =
        (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;
    (-b / (2. * a), ((c - b * b) / (4. * a)))
}

pub fn get_homography_inverse(homography: &Array2<f64>) -> Array2<f64> {
    let nalg_matrix = nalgebra::dmatrix![
        homography[[0, 0]], homography[[0, 1]], homography[[0, 2]];
        homography[[1, 0]], homography[[1, 1]], homography[[1, 2]];
        homography[[2, 0]], homography[[2, 1]], homography[[2, 2]]];
    let inverted = nalg_matrix.try_inverse().unwrap();
    /* gotta transpose it because nalgebra gives the matrix in column-major order
    but ndarray expects it in row-major order */
    Array2::from_shape_vec((3, 3), inverted.transpose().data.as_vec().clone()).unwrap()
}
