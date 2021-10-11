use std::cmp::Ordering;
use std::f32::consts::PI;
use std::sync::mpsc::channel;
use std::thread;

use image::imageops::filter3x3;
use image::DynamicImage;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::RgbImage;

use ndarray::arr2;
use ndarray::arr3;
use ndarray::concatenate;
use ndarray::s;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayView3;
use ndarray::Axis;
use ndarray::Zip;

use show_image::WindowOptions;
use show_image::WindowProxy;

use rand::Rng;

const TWO_PI: f32 = PI * 2.;
const SIFT_WINDOW_SIZE: i64 = 8;
const SIFT_BIN_COUNT: usize = 8;
const SIFT_DESCRIPTOR_LENGTH: usize = 16 * SIFT_BIN_COUNT;
const MAX_KEYPOINTS_PER_IMAGE: usize = 750;
const KP_ORIENTATION_WINDOW_SIZE: i64 = 5;
const GAUSSIAN_SIGMA: f32 = 0.85;

type NDRgbImage = Array3<f32>;
type NDGrayImage = Array3<f32>;
type SiftDescriptor = [f32; SIFT_DESCRIPTOR_LENGTH];

// TODO: use f64 instead of f32?

// cv.getGaussianKernel(9, 1.6)
const GAUSSIAN_BASIC: [f32; 9] = [
    0.011002f32,
    0.04317515f32,
    0.11464352f32,
    0.2059771f32,
    0.25040447f32,
    0.2059771f32,
    0.11464352f32,
    0.04317515f32,
    0.011002f32,
];

const GAUSSIAN_: [f32; 9] = [
    1f32 / 16f32,
    1f32 / 8f32,
    1f32 / 16f32,
    1f32 / 8f32,
    1f32 / 4f32,
    1f32 / 8f32,
    1f32 / 16f32,
    1f32 / 8f32,
    1f32 / 16f32,
];

#[derive(Clone, Copy, Debug)]
struct KeyPoint {
    x: i64,
    y: i64,
    value: f32,
}

#[derive(Clone, Debug)]
struct DescriptedKeyPoint {
    x: i64,
    y: i64,
    value: f32,
    orientation: f32,
    descriptor: SiftDescriptor,
}

struct SplitImage {
    r: NDGrayImage,
    g: NDGrayImage,
    b: NDGrayImage,
}

struct ImageGradients {
    x: NDRgbImage,
    y: NDRgbImage,
}

struct ImageGradientsGray {
    x: NDRgbImage,
    y: NDRgbImage,
}

struct ImageTreeNode {
    image: NDRgbImage,
    key_points: Option<Vec<DescriptedKeyPoint>>,
    children: Option<Vec<ImageTreeNode>>,
}

fn gaussian(x: f32, mu: f32, sigma: f32) -> f32 {
    let a = (x - mu) / sigma;
    (-0.5 * a * a).exp()
}

fn gaussian_kernel_2d(radius: u16, sigma_in: Option<f32>) -> Array2<f32> {
    let sigma = sigma_in.unwrap_or(radius as f32 / 2.);
    let size = 2 * radius as usize + 1;
    let mut kernel = Array2::zeros((size, size));
    let mut sum = 0f32;
    for x in 0..size {
        for y in 0..size {
            let val =
                gaussian(x as f32, radius as f32, sigma) * gaussian(y as f32, radius as f32, sigma);
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

fn show_image(img: DynamicImage, name: &str) -> WindowProxy {
    let window = show_image::create_window(name, Default::default()).unwrap();
    window.set_image(name, img).unwrap();
    // show_image::back
    window
}

fn show_rgb_image(img: RgbImage, name: &str) -> WindowProxy {
    show_image(image::DynamicImage::ImageRgb8(img), name)
}

fn show_grayscale_image(img: GrayImage, name: &str) -> WindowProxy {
    show_image(image::DynamicImage::ImageLuma8(img), name)
}
struct WindowStatus {
    window: WindowProxy,
    key_pressed: bool,
}

fn wait_for_windows_to_close(windows: Vec<WindowProxy>) {
    if windows.len() == 0 {
        return;
    }
    let (tx, rx) = channel();
    for window in &windows {
        let _tx = tx.clone();
        let event_receiver = window.event_channel().unwrap();
        thread::spawn(move || loop {
            let event = event_receiver.recv().unwrap();
            if let show_image::event::WindowEvent::KeyboardInput(event) = event {
                if !event.is_synthetic && event.input.state.is_pressed() {
                    println!("Key pressed!");
                    _tx.send(()).unwrap();
                    break;
                }
            }
        });
    }
    rx.recv().unwrap();
}

fn load_rgb_image_from_file(path: &str) -> RgbImage {
    image::open(path).unwrap().into_rgb8()
}

fn split_img_channels(img: &NDRgbImage) -> SplitImage {
    let r = img.clone().slice_move(s![.., .., 0..1]);
    let g = img.clone().slice_move(s![.., .., 1..2]);
    let b = img.clone().slice_move(s![.., .., 2..3]);
    SplitImage { r, g, b }
}

fn merge_img_channels(img: &SplitImage) -> NDRgbImage {
    concatenate(Axis(2), &[img.r.view(), img.g.view(), img.b.view()]).unwrap()
}

fn rgb_img_to_grayscale(img: &NDRgbImage) -> NDGrayImage {
    let img_shape = img.shape();
    let mut out = Array3::zeros((img_shape[0], img_shape[1], 1));
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            // http://www.songho.ca/dsp/luminance/luminance.html
            out[[x, y, 0]] =
                0.288 * img[[x, y, 0]] + 0.587 * img[[x, y, 1]] + 0.114 * img[[x, y, 2]];
        }
    }
    out
}

fn image_to_ndarray_rgb(img: RgbImage) -> NDRgbImage {
    let mut out = Array3::zeros((img.width() as usize, img.height() as usize, 3));
    for x in 0..img.width() as usize {
        for y in 0..img.height() as usize {
            let pixel = img.get_pixel(x as u32, y as u32);
            out[[x, y, 0]] = pixel[0] as f32;
            out[[x, y, 1]] = pixel[1] as f32;
            out[[x, y, 2]] = pixel[2] as f32;
        }
    }
    out
}

fn image_to_ndarray_gray(img: GrayImage) -> NDRgbImage {
    let mut out = Array3::zeros((img.width() as usize, img.height() as usize, 0));
    for x in 0..img.width() as usize {
        for y in 0..img.height() as usize {
            let pixel = img.get_pixel(x as u32, y as u32);
            out[[x, y, 0]] = pixel[0] as f32;
        }
    }
    out
}

fn ndarray_to_image_rgb(img: NDRgbImage) -> RgbImage {
    let img_shape = img.shape();
    let mut out = RgbImage::new(img_shape[0] as u32, img_shape[1] as u32);
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let clamp = |val: f32| val.max(0f32).min(255f32).round() as u8;
            let r = clamp(img[[x, y, 0]]);
            let g = clamp(img[[x, y, 1]]);
            let b = clamp(img[[x, y, 2]]);
            out.put_pixel(x as u32, y as u32, Rgb::from([r, g, b]));
        }
    }
    out
}

enum ImgConversionType {
    CLAMP,
    NORMALIZE,
}

fn ndarray_to_image_gray(img: NDGrayImage, conversion_type: ImgConversionType) -> GrayImage {
    let img_shape = img.shape();
    let mut out = GrayImage::new(img_shape[0] as u32, img_shape[1] as u32);
    let mut max_val = 0f32;
    if let ImgConversionType::NORMALIZE = conversion_type {
        for x in 0..img_shape[0] {
            for y in 0..img_shape[1] {
                let val = img[[x, y, 0]];
                if max_val < val {
                    max_val = val;
                }
            }
        }
    }
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let convert = |val: f32| match conversion_type {
                ImgConversionType::NORMALIZE => (255f32 * val.max(0f32) / max_val).round() as u8,
                ImgConversionType::CLAMP => val.max(0f32).min(255f32).round() as u8,
            };
            let r = convert(img[[x, y, 0]]);
            out.put_pixel(x as u32, y as u32, Luma::from([r]));
        }
    }
    out
}

fn filter_2d(img: &NDGrayImage, kernel: &Array2<f32>) -> NDGrayImage {
    let img_shape = img.shape();
    let kernel_shape = kernel.shape();
    let mut out = Array3::zeros((img_shape[0], img_shape[1], img_shape[2]));
    let half_kernel_width = ((kernel_shape[0] as f32) / 2.).floor() as i64;
    let half_kernel_height = ((kernel_shape[1] as f32) / 2.).floor() as i64;
    let mut relative_kernel_positions = Array3::zeros((kernel_shape[0], kernel_shape[1], 2));
    for x in 0..kernel_shape[0] {
        for y in 0..kernel_shape[1] {
            relative_kernel_positions[[x, y, 0]] = -half_kernel_width + (x as i64);
            relative_kernel_positions[[x, y, 1]] = -half_kernel_height + (y as i64);
        }
    }
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let mut final_val = 0f32;
            for k_x in 0..kernel_shape[0] {
                for k_y in 0..kernel_shape[1] {
                    let relative_position_x =
                        relative_kernel_positions[[k_x as usize, k_y as usize, 0]];
                    let relative_position_y =
                        relative_kernel_positions[[k_x as usize, k_y as usize, 1]];
                    let p_x = (x as i64 + relative_position_x)
                        .max(0)
                        .min(img_shape[0] as i64 - 1) as usize;
                    let p_y = (y as i64 + relative_position_y)
                        .max(0)
                        .min(img_shape[1] as i64 - 1) as usize;
                    let pixel_value = img[[p_x, p_y, 0]];
                    let kernel_value = kernel[[k_x, k_y]];
                    final_val += pixel_value * kernel_value;
                }
            }
            out[[x, y, 0]] = final_val;
        }
    }
    out
}

fn get_image_gradients(img: &NDRgbImage) -> ImageGradientsGray {
    let sobel_x = Array2::from_shape_vec(
        (3, 3),
        vec![1f32, 0f32, -1f32, 2f32, 0f32, -2f32, 1f32, 0f32, -1f32],
    )
    .unwrap();
    let sobel_y = Array2::from_shape_vec(
        (3, 3),
        vec![1f32, 2f32, 1f32, 0f32, 0f32, 0f32, -1f32, -2f32, -1f32],
    )
    .unwrap();

    let SplitImage { r, g, b } = split_img_channels(img);
    ImageGradientsGray {
        x: rgb_img_to_grayscale(&merge_img_channels(&SplitImage {
            r: filter_2d(&r, &sobel_x),
            g: filter_2d(&g, &sobel_x),
            b: filter_2d(&b, &sobel_x),
        })),
        y: rgb_img_to_grayscale(&merge_img_channels(&SplitImage {
            r: filter_2d(&r, &sobel_y),
            g: filter_2d(&g, &sobel_y),
            b: filter_2d(&b, &sobel_y),
        })),
    }
}

fn get_blended_circle_pixel(
    pixel_color: &[u8; 3],
    pixel_x: i64,
    pixel_y: i64,
    circle_color: &[u8; 3],
    circle_center_x: i64,
    circle_center_y: i64,
    circle_radius: f64,
    stroke_size: f32,
) -> Option<[u8; 3]> {
    let dist_x = circle_center_x - pixel_x;
    let dist_y = circle_center_y - pixel_y;
    let dist = ((dist_x.pow(2) + dist_y.pow(2)) as f64).sqrt();
    let alpha = 0f64.max(1f64 - ((dist - circle_radius).abs() / stroke_size as f64));
    if alpha > 0f64 {
        let blended_colors: Vec<u8> = circle_color
            .iter()
            .enumerate()
            // interpolate from img_portion to keypoint_portion by alpha
            .map(|(i, channel)| {
                let circle_portion = channel.clone() as f64 * alpha;
                let img_portion = pixel_color[i] as f64 * (1f64 - alpha);
                let final_channel = circle_portion + img_portion;
                let final_channel_clamped = final_channel
                    .max(u8::MIN as f64)
                    .min(u8::MAX as f64)
                    .round() as u8;
                final_channel_clamped
            })
            .collect();
        let blended_colors_arr = [blended_colors[0], blended_colors[1], blended_colors[2]];
        return Some(blended_colors_arr);
    }
    None
}

fn draw_key_points(img: &RgbImage, key_points: &Vec<KeyPoint>) -> RgbImage {
    let mut result = RgbImage::new(img.width(), img.height());
    let key_point_colors: Vec<[u8; 3]> = (0..key_points.len())
        .map(|_| [rand::random(), rand::random(), rand::random()])
        .collect();
    for (x, y, _) in img.enumerate_pixels() {
        let img_color = img.get_pixel(x, y);
        result.put_pixel(x, y, img_color.clone());

        for (i, key_point) in key_points.iter().enumerate() {
            if let Some(blended_color) = get_blended_circle_pixel(
                &[img_color[0], img_color[1], img_color[2]],
                x as i64,
                y as i64,
                &key_point_colors[i],
                key_point.x as i64,
                key_point.y as i64,
                12f64,
                1f32,
            ) {
                result.put_pixel(x, y, Rgb::from(blended_color));
            }
        }
    }
    for (i, key_point) in key_points.iter().enumerate() {
        result.put_pixel(
            key_point.x as u32,
            key_point.y as u32,
            Rgb::from(key_point_colors[i]),
        );
    }
    result
}

fn multiply_per_pixel(img1: &NDGrayImage, img2: &NDGrayImage) -> NDGrayImage {
    let img_shape = img1.shape();
    let mut out = Array3::zeros((img_shape[0], img_shape[1], 1));
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            out[[x, y, 0]] = img1[[x, y, 0]] * img2[[x, y, 0]];
        }
    }
    out
}

fn compute_harris_response(img: &NDRgbImage, img_gradients: &ImageGradientsGray) -> NDGrayImage {
    let mut harris_matrices = [
        multiply_per_pixel(&img_gradients.x, &img_gradients.x),
        multiply_per_pixel(&img_gradients.x, &img_gradients.y),
        multiply_per_pixel(&img_gradients.y, &img_gradients.x),
        multiply_per_pixel(&img_gradients.y, &img_gradients.y),
    ];
    let gaussian_kernel = gaussian_kernel_2d(5, Some(GAUSSIAN_SIGMA));
    for i in 0..harris_matrices.len() {
        harris_matrices[i] = filter_2d(&harris_matrices[i], &gaussian_kernel);
    }
    let img_shape = img.shape();
    let mut corner_strengths = Array3::zeros((img_shape[0], img_shape[1], 1));
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let a = harris_matrices[0][[x, y, 0]];
            let b = harris_matrices[1][[x, y, 0]];
            let c = harris_matrices[2][[x, y, 0]];
            let d = harris_matrices[3][[x, y, 0]];
            corner_strengths[[x, y, 0]] = (a * d - b * c) / ((a + d) + 0.00001);
        }
    }
    corner_strengths
}

fn is_max_among_neighbors(img: &NDRgbImage, x: i64, y: i64) -> bool {
    let val = img[[x as usize, y as usize, 0]];
    if val == 0f32 {
        return false;
    }
    let img_shape = img.shape();
    let neighbor_positions: [(i64, i64); 9] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let mut max_among_neighbors = 0f32;
    for (relative_x, relative_y) in neighbor_positions {
        let neighbor_x = (x + relative_x).max(0).min(img_shape[0] as i64 - 1);
        let neighbor_y = (y + relative_y).max(0).min(img_shape[1] as i64 - 1);
        let neighbor_val = img[[neighbor_x as usize, neighbor_y as usize, 0]];
        if neighbor_val > max_among_neighbors {
            max_among_neighbors = neighbor_val;
        }
    }
    val >= max_among_neighbors
}

fn point_near_boundary(x: i64, y: i64, img_width: i64, img_height: i64) -> bool {
    x < SIFT_WINDOW_SIZE + 1
        || y < SIFT_WINDOW_SIZE + 1
        || x > img_width - SIFT_WINDOW_SIZE - 1
        || y > img_height - SIFT_WINDOW_SIZE - 1
}

fn compute_harris_keypoints(img: &NDRgbImage, img_gradients: &ImageGradientsGray) -> Vec<KeyPoint> {
    let corner_strengths = compute_harris_response(img, img_gradients);
    let mut max_corner_strength = 0f32;
    let img_shape = img.shape();
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let corner_strength = corner_strengths[[x, y, 0]];
            if corner_strength > max_corner_strength {
                max_corner_strength = corner_strength;
            }
        }
    }
    let mut corner_strengths_thresholded = Array3::zeros((img_shape[0], img_shape[1], 1));
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let corner_strength = corner_strengths[[x, y, 0]];
            if corner_strength > max_corner_strength * 0.2f32 {
                corner_strengths_thresholded[[x, y, 0]] = corner_strengths[[x, y, 0]];
            } else {
                corner_strengths_thresholded[[x, y, 0]] = 0f32;
            }
        }
    }
    let mut key_points: Vec<KeyPoint> = Vec::new();
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let value = corner_strengths_thresholded[[x, y, 0]];
            if is_max_among_neighbors(&corner_strengths_thresholded, x as i64, y as i64)
                && value > 0f32
                && !point_near_boundary(
                    x as i64,
                    y as i64,
                    img_shape[0] as i64,
                    img_shape[1] as i64,
                )
            {
                key_points.push(KeyPoint {
                    x: x as i64,
                    y: y as i64,
                    value,
                });
            }
        }
    }
    key_points.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal));
    let max_keypoint = MAX_KEYPOINTS_PER_IMAGE.min(key_points.len() - 1);
    let key_points_sliced = (&key_points[..max_keypoint]).to_vec();
    println!("Found {:?} keypoints", key_points_sliced.len());
    key_points_sliced
}

fn get_parabola_vertex(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> (f32, f32) {
    let denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
    let a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
    let b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom;
    let c =
        (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;
    (-b / (2. * a), ((c - b * b) / (4. * a)))
}

fn get_key_point_orientations(
    key_point: &KeyPoint,
    gradient_magnitudes: &Array3<f32>,
    gradient_directions: &Array3<f32>,
) -> Vec<f32> {
    let img_shape = gradient_magnitudes.shape();

    let gaussian_kernel =
        gaussian_kernel_2d(KP_ORIENTATION_WINDOW_SIZE as u16, Some(GAUSSIAN_SIGMA));
    let neighborhood = (
        (
            (key_point.x - KP_ORIENTATION_WINDOW_SIZE).max(0),
            (key_point.y - KP_ORIENTATION_WINDOW_SIZE).max(0),
        ),
        (
            (key_point.x + KP_ORIENTATION_WINDOW_SIZE + 1).min(img_shape[0] as i64),
            (key_point.y + KP_ORIENTATION_WINDOW_SIZE + 1).min(img_shape[1] as i64),
        ),
    );

    // println!(
    //     "yope 1 img_shape={:?}, neighhb={:?}",
    //     img_shape, neighborhood
    // );
    const BIN_COUNT: usize = 10;
    let mut orientation_votes = [0 as f32; BIN_COUNT];
    for x in (neighborhood.0 .0)..(neighborhood.1 .0) {
        for y in (neighborhood.0 .1)..(neighborhood.1 .1) {
            // println!(
            //     "x={}, y={}, key_point.x={}, key_point.y={}, f_y={:?}",
            //     x,
            //     y,
            //     key_point.x,
            //     key_point.y,
            //     (y - key_point.y + KP_ORIENTATION_WINDOW_SIZE) as usize,
            // );
            let gradient_direction =
                ((gradient_directions[[x as usize, y as usize, 0]] + TWO_PI) % TWO_PI) / (TWO_PI);
            let bin_index = (BIN_COUNT as f32 * gradient_direction).floor() as usize;
            let gaussian_factor = gaussian_kernel[[
                (x - key_point.x + KP_ORIENTATION_WINDOW_SIZE) as usize,
                (y - key_point.y + KP_ORIENTATION_WINDOW_SIZE) as usize,
            ]];
            let bin_value = gaussian_factor * gradient_magnitudes[[x as usize, y as usize, 0]];
            orientation_votes[bin_index] += bin_value;
        }
    }
    let dominant_bin_value =
        orientation_votes
            .iter()
            .fold(0f32, |acc, val| if val > &acc { *val } else { acc });
    let peak_bin_indices: Vec<usize> = orientation_votes
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, val)| *val > dominant_bin_value * 0.8f32)
        .map(|(i, _)| i)
        .collect();
    peak_bin_indices
        .iter()
        .cloned()
        .map(|peak_bin_index| {
            let left_index = peak_bin_index - 1;
            let right_index = peak_bin_index + 2;
            let parabolic_average_index = get_parabola_vertex(
                left_index as f32,
                orientation_votes[(left_index + BIN_COUNT) % BIN_COUNT],
                peak_bin_index as f32,
                orientation_votes[peak_bin_index],
                right_index as f32,
                orientation_votes[(peak_bin_index + 2) % BIN_COUNT],
            )
            .0;
            ((parabolic_average_index / BIN_COUNT as f32) % BIN_COUNT as f32) * TWO_PI
        })
        .collect()
}

fn get_key_point_descriptor(
    key_point: &KeyPoint,
    gradient_magnitudes: &Array3<f32>,
    gradient_directions: &Array3<f32>,
    key_point_orientation: f32,
) -> SiftDescriptor {
    let img_shape = gradient_magnitudes.shape();
    let gaussian_kernel = gaussian_kernel_2d(SIFT_WINDOW_SIZE as u16, Some(GAUSSIAN_SIGMA));
    let neighborhood = (
        (
            (key_point.x - SIFT_WINDOW_SIZE),
            (key_point.y - SIFT_WINDOW_SIZE),
        ),
        (
            (key_point.x + SIFT_WINDOW_SIZE),
            (key_point.y + SIFT_WINDOW_SIZE),
        ),
    );
    let relative_gradient_directions =
        gradient_directions.map(|direction| (direction - key_point_orientation + TWO_PI) % TWO_PI);

    // println!(
    //     "yope 1 img_shape={:?}, neighhb={:?}",
    //     img_shape, neighborhood
    // );
    let mut descriptor_2: SiftDescriptor = [0 as f32; SIFT_DESCRIPTOR_LENGTH];
    // let mut descriptor: Vec<f32> = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            let start_x = neighborhood.0 .0 + (i * 4);
            let start_y = neighborhood.0 .1 + (j * 4);
            let sub_neighborhood = ((start_x, start_y), (start_x + 4, start_y + 4));

            // println!("subneighhb={:?}", sub_neighborhood);

            // let mut orientation_vote_counts = [0 as f32; SIFT_BIN_COUNT];
            for x in (sub_neighborhood.0 .0)..(sub_neighborhood.1 .0) {
                for y in (sub_neighborhood.0 .1)..(sub_neighborhood.1 .1) {
                    // println!(
                    //     "x={}, y={}, key_point.x={}, key_point.y={}, f_y={:?}",
                    //     x,
                    //     y,
                    //     key_point.x,
                    //     key_point.y,
                    //     (y - key_point.y + SIFT_WINDOW_SIZE) as usize,
                    // );
                    let bin_index = (SIFT_BIN_COUNT as f32
                        * relative_gradient_directions[[x as usize, y as usize, 0]]
                        / TWO_PI)
                        .floor() as usize;
                    let gaussian_factor = gaussian_kernel[[
                        (x - key_point.x + SIFT_WINDOW_SIZE) as usize,
                        (y - key_point.y + SIFT_WINDOW_SIZE) as usize,
                    ]];
                    // orientation_vote_counts[bin_index] +=
                    //     gaussian_factor * gradient_magnitudes[[x as usize, y as usize, 0]];
                    let descriptor_index = ((i * 4 + j) * SIFT_BIN_COUNT as i64) + bin_index as i64;
                    // println!("descriptor_index={:?}", descriptor_index);
                    descriptor_2[descriptor_index as usize] +=
                        gaussian_factor * gradient_magnitudes[[x as usize, y as usize, 0]];
                }
            }
            // descriptor.append(&mut orientation_vote_counts.to_vec());
        }
    }

    // panic!();

    // println!(
    //     "descriptor len={:?}, descriptor_2 len={:?}",
    //     descriptor.len(),
    //     descriptor_2.len()
    // );

    // println!(
    //     "descriptor={:?}, descriptor_2={:?}",
    //     descriptor, descriptor_2
    // );

    let mut descriptors_are_equal = true;

    // for (i, val) in descriptor.iter().enumerate() {
    //     if *val != descriptor_2[i] {
    //         descriptors_are_equal = false;
    //     }
    // }

    if !descriptors_are_equal {
        println!(
            "descriptors_are_equal={:?}!!!!!!!!!!!!",
            descriptors_are_equal
        );
    }

    let final_descriptor = normalize_sift_descriptor(&mut clip_sift_descriptor(
        &mut normalize_sift_descriptor(&mut descriptor_2),
        0.2f32,
    ));

    final_descriptor
}

fn normalize_vec(vec: &Vec<f32>) -> Vec<f32> {
    let norm = vec.iter().fold(0f32, |acc, val| acc + val * val).sqrt();
    vec.iter().cloned().map(|val| val / norm).collect()
}

fn normalize_sift_descriptor(arr: &mut SiftDescriptor) -> SiftDescriptor {
    let mut total_of_squares = 0f32;
    for i in 0..arr.len() {
        total_of_squares += arr[i] * arr[i];
    }
    let total_of_squares_sqrt = total_of_squares.sqrt();
    for i in 0..arr.len() {
        arr[i] /= total_of_squares_sqrt;
    }
    *arr
}

fn clip_sift_descriptor(arr: &mut SiftDescriptor, max: f32) -> SiftDescriptor {
    for i in 0..arr.len() {
        if arr[i] > max {
            arr[i] = max
        }
    }
    *arr
}

fn get_keypoint_descriptors(
    img: &NDRgbImage,
    img_gradients: &ImageGradientsGray,
    key_points: &Vec<KeyPoint>,
) -> Vec<DescriptedKeyPoint> {
    let img_shape = img.shape();
    let mut gradient_magnitudes = Array3::zeros((img_shape[0], img_shape[1], 1));
    let mut gradient_directions = Array3::zeros((img_shape[0], img_shape[1], 1));
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            gradient_magnitudes[[x, y, 0]] = (img_gradients.x[[x, y, 0]]
                * img_gradients.x[[x, y, 0]]
                + img_gradients.y[[x, y, 0]] * img_gradients.y[[x, y, 0]])
            .sqrt();
            gradient_directions[[x, y, 0]] =
                (img_gradients.y[[x, y, 0]]).atan2(img_gradients.x[[x, y, 0]]) + PI;
        }
    }
    let mut count = 0;
    let mut descripted_key_points: Vec<DescriptedKeyPoint> = Vec::new();
    for key_point in key_points {
        let orientations =
            get_key_point_orientations(&key_point, &gradient_magnitudes, &gradient_directions);
        let descriptors: Vec<SiftDescriptor> = orientations
            .iter()
            .cloned()
            .map(|orientation| {
                get_key_point_descriptor(
                    &key_point,
                    &gradient_magnitudes,
                    &gradient_directions,
                    orientation,
                )
            })
            .collect();
        let mut new_key_points: Vec<DescriptedKeyPoint> = descriptors
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, descriptor)| DescriptedKeyPoint {
                x: key_point.x,
                y: key_point.y,
                value: key_point.value,
                orientation: orientations[i],
                descriptor,
            })
            .collect();
        count += 1;
        if count % 100 == 0 {
            println!("Done {:?} / {:?} keypoints", count, key_points.len());
        }
        descripted_key_points.append(&mut new_key_points);
    }
    println!("New keypoint count: {:?}", descripted_key_points.len());
    descripted_key_points
}

fn do_sift(image_tree_node: &mut ImageTreeNode) {
    if image_tree_node.key_points.is_none() {
        println!("-- Harris keypoints --");
        let img_gradient = get_image_gradients(&image_tree_node.image);
        let img_harris_keypoints = compute_harris_keypoints(&image_tree_node.image, &img_gradient);
        println!("-- SIFT Descriptors --");
        let img_descripted_keypoints =
            get_keypoint_descriptors(&image_tree_node.image, &img_gradient, &img_harris_keypoints);
        image_tree_node.key_points = Some(img_descripted_keypoints);
    } else {
        println!("Keypoints already computed. Skipping this step");
    }
}

#[show_image::main]
fn main() {
    let mut windows: Vec<WindowProxy> = Vec::new();

    let building_img =
        load_rgb_image_from_file("../featureMatcher/image_sets/panorama/pano1_0008.png");
    // let pixel = building_img.get_pixel(building_img.width() - 1, building_img.height() - 1);
    // println!("{}, {}, {}", pixel[0], pixel[1], pixel[2]);

    // let image_file_result = image::open("../featureMatcher/image_sets/project_images/Rainier1.png");
    // let image_file = image_file_result.unwrap();
    // image_file.into_rgb8();
    // println!(
    //     "{:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
    //     image_file.as_rgb8(),
    //     image_file.as_rgb16(),
    //     image_file.as_rgba8(),
    //     image_file.as_rgba16(),
    //     image_file.as_luma8(),
    //     image_file.as_luma16(),
    //     image_file.as_bgr8(),
    // );

    let rainier_1_image =
        load_rgb_image_from_file("../featureMatcher/image_sets/project_images/Rainier1.png");
    let rainier_2_image =
        load_rgb_image_from_file("../featureMatcher/image_sets/project_images/Rainier2.png");

    let mut rainier_image_tree = ImageTreeNode {
        image: image_to_ndarray_rgb(rainier_1_image),
        children: Some(vec![ImageTreeNode {
            image: image_to_ndarray_rgb(rainier_2_image),
            children: None,
            key_points: None,
        }]),
        key_points: None,
    };

    do_sift(&mut rainier_image_tree);

    // let building_img_w_key_points = draw_key_points(
    //     &building_img,
    //     &vec![
    //         KeyPoint {
    //             x: 10,
    //             y: 10,
    //         },
    //         KeyPoint {
    //             x: 100,
    //             y: 10,
    //         },
    //         KeyPoint {
    //             x: 100,
    //             y: 100,
    //         },
    //         KeyPoint {
    //             x: 10,
    //             y: 100,
    //         },
    //     ],
    // );

    // let SplitImage {
    //     g: building_img_g,
    //     b: building_img_b,
    //     ..
    // } = split_img_channels(&building_img);
    // let ImageGradients {
    //     x: building_img_g_grad_x,
    //     y: building_img_g_grad_y,
    // } = get_image_gradients(&building_img_g);

    // let kernel = gaussian_kernel_2d(1, Some(GAUSSIAN_SIGMA));
    // let kernel2 = gaussian_kernel_2d(2, None);

    // println!("{:?}", kernel);
    // println!("{:?}", kernel2);

    // println!("0.1");
    let rainier_image =
        load_rgb_image_from_file("../featureMatcher/image_sets/project_images/Rainier1.png");
    let rainier_image_nd = image_to_ndarray_rgb(rainier_image.clone());
    // let rainier_image_split = split_img_channels(&rainier_image_nd);
    // let gaussian_kernel = gaussian_kernel_2d(1, Some(GAUSSIAN_SIGMA));
    // let rainier_image_split_blurred = SplitImage {
    //     r: filter_2d(&rainier_image_split.r, &gaussian_kernel),
    //     g: filter_2d(&rainier_image_split.g, &gaussian_kernel),
    //     b: filter_2d(&rainier_image_split.b, &gaussian_kernel),
    // };
    // let rainier_image_nd_blurred = merge_img_channels(&rainier_image_split_blurred);

    // windows.push(show_rgb_image(rainier_image.clone(), "rainier_image"));
    // windows.push(show_rgb_image(
    //     ndarray_to_image_rgb(rainier_image_nd_blurred.clone()),
    //     "rainier_image_nd_blurred",
    // ));

    // let rainier_image_conv = ndarray_to_image_rgb(rainier_image_nd.clone());
    // let ImageGradientsGray {
    //     x: rainier_grad_x,
    //     y: rainier_grad_y,
    // } = get_image_gradients(&rainier_image_nd);
    // println!("0.5");
    // let rainier_grad_x_img =
    //     ndarray_to_image_gray(rainier_grad_x.clone(), ImgConversionType::NORMALIZE);
    // let gaussian_blur_kernel = Array2::from_shape_vec((3, 3), GAUSSIAN_BASIC.to_vec()).unwrap();
    // let rainier_grad_x_blurred_img = ndarray_to_image_gray(
    //     filter_2d(&rainier_grad_x, &gaussian_blur_kernel),
    //     ImgConversionType::NORMALIZE,
    // );
    // println!("0.6");
    // let rainier_grad_y_img =
    //     ndarray_to_image_gray(rainier_grad_y.clone(), ImgConversionType::NORMALIZE);

    println!("1");
    let img_gradient = get_image_gradients(&rainier_image_nd);
    println!("2");
    let img_harris_keypoints = compute_harris_keypoints(&rainier_image_nd, &img_gradient);
    println!("3");
    let rainier_annotated = draw_key_points(&rainier_image, &img_harris_keypoints);
    let kps: Vec<KeyPoint> = rainier_image_tree
        .key_points
        .unwrap()
        .iter()
        .cloned()
        .map(|key_point| KeyPoint {
            x: key_point.x,
            y: key_point.y,
            value: key_point.value,
        })
        .collect();
    let rainier_annotated_2 = draw_key_points(&rainier_image, &kps);

    // println!("Opening windows");
    // windows.push(show_rgb_image(rainier_image.clone(), "rainier_image"));
    // windows.push(show_rgb_image(
    //     rainier_image_conv.clone(),
    //     "rainier_image_conv",
    // ));
    // windows.push(show_grayscale_image(
    //     rainier_grad_x_img.clone(),
    //     "rainier_grad_x_img",
    // ));
    // windows.push(show_grayscale_image(
    //     rainier_grad_x_blurred_img.clone(),
    //     "rainier_grad_x_blurred_img",
    // ));
    // windows.push(show_grayscale_image(
    //     rainier_grad_y_img.clone(),
    //     "rainier_grad_y_img",
    // ));
    windows.push(show_rgb_image(
        rainier_annotated.clone(),
        "rainier_annotated",
    ));
    windows.push(show_rgb_image(
        rainier_annotated_2.clone(),
        "rainier_annotated_2",
    ));

    // println!("Windows opened");

    wait_for_windows_to_close(windows);
}
