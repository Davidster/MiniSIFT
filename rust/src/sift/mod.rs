use std::cmp::Ordering;
use std::f64::consts::PI;

use ndarray::Array2;
use ndarray::Array3;

use super::helpers::*;
use super::image_helpers::*;

const TWO_PI: f64 = PI * 2.;
const SIFT_WINDOW_SIZE: i64 = 8;
const SIFT_BIN_COUNT: usize = 8;
const SIFT_DESCRIPTOR_LENGTH: usize = 16 * SIFT_BIN_COUNT;
const MAX_KEYPOINTS_PER_IMAGE: usize = 750;
const KP_ORIENTATION_WINDOW_SIZE: i64 = 5;
const GAUSSIAN_SIGMA: f64 = 0.85;

type SiftDescriptor = [f64; SIFT_DESCRIPTOR_LENGTH];

#[derive(Clone, Copy, Debug)]
pub struct KeyPoint {
    pub x: i64,
    pub y: i64,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct DescriptedKeyPoint {
    pub x: i64,
    pub y: i64,
    pub value: f64,
    pub orientation: f64,
    pub descriptor: SiftDescriptor,
}

pub struct ImageTreeNode {
    pub image: NDRgbImage,
    pub key_points: Option<Vec<DescriptedKeyPoint>>,
    pub children: Option<Vec<ImageTreeNode>>,
}

pub fn filter_2d(img: &NDGrayImage, kernel: &Array2<f64>) -> NDGrayImage {
    let img_shape = img.shape();
    let kernel_shape = kernel.shape();
    let mut out = Array3::zeros((img_shape[0], img_shape[1], img_shape[2]));
    let half_kernel_width = ((kernel_shape[0] as f64) / 2.).floor() as i64;
    let half_kernel_height = ((kernel_shape[1] as f64) / 2.).floor() as i64;
    let mut relative_kernel_positions = Array3::zeros((kernel_shape[0], kernel_shape[1], 2));
    for x in 0..kernel_shape[0] {
        for y in 0..kernel_shape[1] {
            relative_kernel_positions[[x, y, 0]] = -half_kernel_width + (x as i64);
            relative_kernel_positions[[x, y, 1]] = -half_kernel_height + (y as i64);
        }
    }
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let mut final_val = 0.;
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

pub fn get_image_gradients(img: &NDRgbImage) -> ImageGradientsGray {
    let sobel_x =
        Array2::from_shape_vec((3, 3), vec![1., 0., -1., 2., 0., -2., 1., 0., -1.]).unwrap();
    let sobel_y =
        Array2::from_shape_vec((3, 3), vec![1., 2., 1., 0., 0., 0., -1., -2., -1.]).unwrap();

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

pub fn compute_harris_response(
    img: &NDRgbImage,
    img_gradients: &ImageGradientsGray,
) -> NDGrayImage {
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

pub fn is_max_among_neighbors(img: &NDRgbImage, x: i64, y: i64) -> bool {
    let val = img[[x as usize, y as usize, 0]];
    if val == 0. {
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
    let mut max_among_neighbors = 0.;
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

pub fn point_near_boundary(x: i64, y: i64, img_width: i64, img_height: i64) -> bool {
    x < SIFT_WINDOW_SIZE + 1
        || y < SIFT_WINDOW_SIZE + 1
        || x > img_width - SIFT_WINDOW_SIZE - 1
        || y > img_height - SIFT_WINDOW_SIZE - 1
}

pub fn compute_harris_keypoints(
    img: &NDRgbImage,
    img_gradients: &ImageGradientsGray,
) -> Vec<KeyPoint> {
    let corner_strengths = compute_harris_response(img, img_gradients);
    let mut max_corner_strength = 0.;
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
            if corner_strength > max_corner_strength * 0.2 {
                corner_strengths_thresholded[[x, y, 0]] = corner_strengths[[x, y, 0]];
            } else {
                corner_strengths_thresholded[[x, y, 0]] = 0.;
            }
        }
    }
    let mut key_points: Vec<KeyPoint> = Vec::new();
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let value = corner_strengths_thresholded[[x, y, 0]];
            if is_max_among_neighbors(&corner_strengths_thresholded, x as i64, y as i64)
                && value > 0.
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

pub fn get_key_point_orientations(
    key_point: &KeyPoint,
    gradient_magnitudes: &Array3<f64>,
    gradient_directions: &Array3<f64>,
) -> Vec<f64> {
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
    let mut orientation_votes = [0 as f64; BIN_COUNT];
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
            let bin_index = (BIN_COUNT as f64 * gradient_direction).floor() as usize;
            let gaussian_factor = gaussian_kernel[[
                (x - key_point.x + KP_ORIENTATION_WINDOW_SIZE) as usize,
                (y - key_point.y + KP_ORIENTATION_WINDOW_SIZE) as usize,
            ]];
            let bin_value = gaussian_factor * gradient_magnitudes[[x as usize, y as usize, 0]];
            orientation_votes[bin_index] += bin_value;
        }
    }
    let dominant_bin_value = orientation_votes
        .iter()
        .fold(0., |acc, val| if val > &acc { *val } else { acc });
    let peak_bin_indices: Vec<usize> = orientation_votes
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, val)| *val > dominant_bin_value * 0.8)
        .map(|(i, _)| i)
        .collect();
    peak_bin_indices
        .iter()
        .cloned()
        .map(|peak_bin_index| {
            let left_index = peak_bin_index - 1;
            let right_index = peak_bin_index + 2;
            let parabolic_average_index = get_parabola_vertex(
                left_index as f64,
                orientation_votes[(left_index + BIN_COUNT) % BIN_COUNT],
                peak_bin_index as f64,
                orientation_votes[peak_bin_index],
                right_index as f64,
                orientation_votes[(peak_bin_index + 2) % BIN_COUNT],
            )
            .0;
            ((parabolic_average_index / BIN_COUNT as f64) % BIN_COUNT as f64) * TWO_PI
        })
        .collect()
}

pub fn get_key_point_descriptor(
    key_point: &KeyPoint,
    gradient_magnitudes: &Array3<f64>,
    gradient_directions: &Array3<f64>,
    key_point_orientation: f64,
) -> SiftDescriptor {
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
    let mut descriptor_2: SiftDescriptor = [0 as f64; SIFT_DESCRIPTOR_LENGTH];
    // let mut descriptor: Vec<f64> = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            let start_x = neighborhood.0 .0 + (i * 4);
            let start_y = neighborhood.0 .1 + (j * 4);
            let sub_neighborhood = ((start_x, start_y), (start_x + 4, start_y + 4));

            // println!("subneighhb={:?}", sub_neighborhood);

            // let mut orientation_vote_counts = [0 as f64; SIFT_BIN_COUNT];
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
                    let bin_index = (SIFT_BIN_COUNT as f64
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

    // let mut descriptors_are_equal = true;

    // for (i, val) in descriptor.iter().enumerate() {
    //     if *val != descriptor_2[i] {
    //         descriptors_are_equal = false;
    //     }
    // }

    // if !descriptors_are_equal {
    //     println!(
    //         "descriptors_are_equal={:?}!!!!!!!!!!!!",
    //         descriptors_are_equal
    //     );
    // }

    let final_descriptor = normalize_sift_descriptor(&mut clip_sift_descriptor(
        &mut normalize_sift_descriptor(&mut descriptor_2),
        0.2,
    ));

    final_descriptor
}

#[allow(dead_code)]
pub fn normalize_vec(vec: &Vec<f64>) -> Vec<f64> {
    let norm = vec.iter().fold(0., |acc, val| acc + val * val).sqrt();
    vec.iter().cloned().map(|val| val / norm).collect()
}

pub fn normalize_sift_descriptor(arr: &mut SiftDescriptor) -> SiftDescriptor {
    let mut total_of_squares = 0.;
    for i in 0..arr.len() {
        total_of_squares += arr[i] * arr[i];
    }
    let total_of_squares_sqrt = total_of_squares.sqrt();
    for i in 0..arr.len() {
        arr[i] /= total_of_squares_sqrt;
    }
    *arr
}

pub fn clip_sift_descriptor(arr: &mut SiftDescriptor, max: f64) -> SiftDescriptor {
    for i in 0..arr.len() {
        if arr[i] > max {
            arr[i] = max
        }
    }
    *arr
}

pub fn get_keypoint_descriptors(
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

#[allow(dead_code)]
pub fn get_sorted_key_point_pairs(
    _img_1_key_points: &Vec<DescriptedKeyPoint>,
    _img_2_key_points: &Vec<DescriptedKeyPoint>,
) {
    // TODO:
}

pub fn do_sift(image_tree_node: &mut ImageTreeNode) {
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
