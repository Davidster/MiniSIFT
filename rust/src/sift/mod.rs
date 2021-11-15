use std::cmp::Ordering;
use std::collections::HashSet;
use std::f64::consts::PI;

use ndarray::Array2;
use ndarray::Array3;

use statrs::function::factorial::factorial;

use itertools::Itertools;

use opencv::prelude::*;

use super::helpers::*;
use super::image_helpers::*;

const TWO_PI: f64 = PI * 2.;
const SIFT_WINDOW_SIZE: i64 = 8;
const SIFT_BIN_COUNT: usize = 8;
const SIFT_DESCRIPTOR_LENGTH: usize = 16 * SIFT_BIN_COUNT;
const SIFT_INLIER_THRESHOLD: f64 = 5.;
const MAX_KEYPOINTS_PER_IMAGE: usize = 750;
const KP_ORIENTATION_WINDOW_SIZE: i64 = 5;
const GAUSSIAN_SIGMA: f64 = 4.;

type SiftDescriptor = [f64; SIFT_DESCRIPTOR_LENGTH];

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Point {
    pub x: i64,
    pub y: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pointf {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct KeyPoint {
    pub x: i64,
    pub y: i64,
    pub value: f64,
    pub color: Option<[u8; 3]>,
}

#[derive(Clone, Debug)]
pub struct DescriptedKeyPoint {
    pub x: i64,
    pub y: i64,
    pub value: f64,
    pub orientation: f64,
    pub descriptor: SiftDescriptor,
    pub color: Option<[u8; 3]>,
}

pub struct DescriptedImage {
    pub image: NDRgbImage,
    pub key_points: Vec<DescriptedKeyPoint>,
    pub homography: Option<Array2<f64>>,
}

pub fn point_vec_to_cv_mat(points: Vec<Point>) -> opencv::core::Mat {
    let mut res: opencv::core::Mat = unsafe {
        opencv::core::Mat::new_rows_cols(points.len() as i32, 2, opencv::core::CV_32FC2).unwrap()
    };
    opencv::core::Mat::from_slice_2d(
        points
            .iter()
            .map(|Point { x, y }| [*x as f32, *y as f32])
            .collect::<Vec<[f32; 2]>>()
            .as_slice(),
    )
    .unwrap()
    .assign_to(&mut res, opencv::core::CV_32FC2)
    .unwrap();
    res
}

pub fn find_homography(img_1_points: Vec<Point>, img_2_points: Vec<Point>) -> Array2<f64> {
    let img_1_points_cv = point_vec_to_cv_mat(img_1_points);
    let img_2_points_cv = point_vec_to_cv_mat(img_2_points);
    // TODO: why does a 0x0 matrix work here?
    let mut mask: Mat = unsafe { Mat::new_rows_cols(0, 0, opencv::core::CV_32FC2).unwrap() };
    let h_cv =
        opencv::calib3d::find_homography(&img_1_points_cv, &img_2_points_cv, &mut mask, 0, 3.)
            .unwrap();
    let out: Array2<f64> =
        Array2::from_shape_vec((3, 3), h_cv.data_typed::<f64>().unwrap().to_vec()).unwrap();
    out
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
                    color: None,
                });
            }
        }
    }
    key_points.sort_unstable_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal));
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
                (gradient_directions[[x as usize, y as usize, 0]] + TWO_PI) % TWO_PI;
            let bin_index = (BIN_COUNT as f64 * (gradient_direction / TWO_PI)).floor() as usize;
            let gaussian_factor = gaussian_kernel[[
                (x - key_point.x + KP_ORIENTATION_WINDOW_SIZE) as usize,
                (y - key_point.y + KP_ORIENTATION_WINDOW_SIZE) as usize,
            ]];
            let bin_value = gaussian_factor * gradient_magnitudes[[x as usize, y as usize, 0]];
            orientation_votes[bin_index] += bin_value;
        }
    }
    // println!("orientation_votes={:?}", orientation_votes);
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
            let right_index = peak_bin_index + 1;
            let parabolic_average_index = get_parabola_vertex(
                left_index as f64,
                orientation_votes[(left_index + BIN_COUNT) % BIN_COUNT],
                peak_bin_index as f64,
                orientation_votes[peak_bin_index],
                right_index as f64,
                orientation_votes[(right_index + BIN_COUNT) % BIN_COUNT],
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
    // println!("heyyyyyyyyyyyyy");
    let gaussian_kernel = gaussian_kernel_2d(SIFT_WINDOW_SIZE as u16, Some(GAUSSIAN_SIGMA));
    // println!("gaussian_kernel={:?}", gaussian_kernel);
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
    let mut descriptor: SiftDescriptor = [0 as f64; SIFT_DESCRIPTOR_LENGTH];
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
                    // println!("i={:?}, j={:?}, SIFT_BIN_COUNT={:?}, bin_index={:?} -> descriptor_index={:?}", i, j, SIFT_BIN_COUNT, bin_index, descriptor_index);
                    // println!("descriptor_index={:?}", descriptor_index);
                    descriptor[descriptor_index as usize] +=
                        gaussian_factor * gradient_magnitudes[[x as usize, y as usize, 0]];
                }
            }
            // descriptor.append(&mut orientation_vote_counts.to_vec());
        }
    }

    // println!("descriptor.len()={:?}", descriptor);

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
        &mut normalize_sift_descriptor(&mut descriptor),
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
    for key_point in key_points.iter() {
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
                color: key_point.color,
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

pub fn get_sorted_key_point_pairs<'a>(
    img_1_key_points: &'a Vec<DescriptedKeyPoint>,
    img_2_key_points: &'a Vec<DescriptedKeyPoint>,
) -> Vec<(&'a DescriptedKeyPoint, &'a DescriptedKeyPoint, f64)> {
    let mut all_pairs: Vec<(&DescriptedKeyPoint, &DescriptedKeyPoint, f64)> = Vec::new();
    for img_1_key_point in img_1_key_points {
        for img_2_key_point in img_2_key_points {
            let mut sum = 0.;
            for i in 0..img_1_key_point.descriptor.len() {
                let diff = img_2_key_point.descriptor[i] - img_1_key_point.descriptor[i];
                sum += diff * diff;
            }
            all_pairs.push((&img_1_key_point, &img_2_key_point, sum.sqrt()));
        }
    }
    all_pairs.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
    println!(
        "Computed and sorted {:?} keypoint distances",
        all_pairs.len()
    );
    // println!("{:?}", all_pairs[0]);
    // let first_10: Vec<f64> = all_pairs[0..10]
    //     .to_vec()
    //     .iter()
    //     .map(|(_, _, distance)| *distance)
    //     .collect();
    // let last_10: Vec<f64> = all_pairs[all_pairs.len() - 10..]
    //     .to_vec()
    //     .iter()
    //     .map(|(_, _, distance)| *distance)
    //     .collect();
    // println!("First 10: {:?}", first_10);
    // println!("Last 10: {:?}", last_10);
    all_pairs
}

pub fn get_best_matches_by_ratio_test(
    img_1_key_points: &Vec<DescriptedKeyPoint>,
    img_2_key_points: &Vec<DescriptedKeyPoint>,
    sorted_key_point_pairs: Vec<(&DescriptedKeyPoint, &DescriptedKeyPoint, f64)>,
) -> Vec<(DescriptedKeyPoint, DescriptedKeyPoint, f64)> {
    let mut matches: Vec<(&DescriptedKeyPoint, &DescriptedKeyPoint, f64)> = Vec::new();
    for key_point in img_1_key_points {
        let top_two_matches = sorted_key_point_pairs
            .iter()
            .filter(|(img_1_key_point, _, _)| {
                img_1_key_point.x == key_point.x && img_1_key_point.y == key_point.y
            })
            .take(2)
            .collect::<Vec<&(&DescriptedKeyPoint, &DescriptedKeyPoint, f64)>>();
        matches.push((
            top_two_matches[0].0,
            top_two_matches[0].1,
            top_two_matches[0].2 / top_two_matches[1].2,
        ));
    }
    for key_point in img_2_key_points {
        let top_two_matches = sorted_key_point_pairs
            .iter()
            .filter(|(_, img_2_key_point, _)| {
                img_2_key_point.x == key_point.x && img_2_key_point.y == key_point.y
            })
            .take(2)
            .collect::<Vec<&(&DescriptedKeyPoint, &DescriptedKeyPoint, f64)>>();
        matches.push((
            top_two_matches[0].0,
            top_two_matches[0].1,
            top_two_matches[0].2 / top_two_matches[1].2,
        ));
    }
    matches.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
    let filtered_matches: Vec<(&DescriptedKeyPoint, &DescriptedKeyPoint, f64)> = matches
        .iter()
        .cloned()
        .filter(|(_, _, ratio)| *ratio > 0.85)
        .collect();
    // println!("pre dedupe={:?}", filtered_matches.len());
    let mut deduped_matches = dedupe_points(filtered_matches);
    println!(
        "{:?} matched points remain after ratio test",
        deduped_matches.len()
    );
    for mut deduped_match in &mut deduped_matches {
        let color = [rand::random(), rand::random(), rand::random()];
        deduped_match.0.color = Some(color);
        deduped_match.1.color = Some(color);
    }
    deduped_matches
}

pub fn dedupe_points<'a>(
    key_point_pairs: Vec<(&'a DescriptedKeyPoint, &'a DescriptedKeyPoint, f64)>,
) -> Vec<(DescriptedKeyPoint, DescriptedKeyPoint, f64)> {
    let mut used_points: HashSet<Point> = HashSet::new();
    let mut deduped: Vec<(DescriptedKeyPoint, DescriptedKeyPoint, f64)> = Vec::new();
    for pair in key_point_pairs {
        let img_1_point = Point {
            x: pair.0.x,
            y: pair.0.y,
        };
        let img_2_point = Point {
            x: pair.1.x,
            y: pair.1.y,
        };
        if used_points.contains(&img_1_point) || used_points.contains(&img_2_point) {
            continue;
        }
        used_points.insert(img_1_point);
        used_points.insert(img_2_point);
        deduped.push((pair.0.clone(), pair.1.clone(), pair.2));
    }
    deduped
}

pub fn do_sift(image: &NDRgbImage) -> Vec<DescriptedKeyPoint> {
    println!("-- Harris keypoints --");
    let img_gradient = get_image_gradients(&image);
    let img_harris_keypoints = compute_harris_keypoints(&image, &img_gradient);
    println!("-- SIFT Descriptors --");
    get_keypoint_descriptors(&image, &img_gradient, &img_harris_keypoints)
}

pub fn do_ransac(
    matched_key_points: Vec<(DescriptedKeyPoint, DescriptedKeyPoint, f64)>,
) -> (
    Array2<f64>,
    Vec<(DescriptedKeyPoint, DescriptedKeyPoint, f64)>,
) {
    let point_count = matched_key_points.len();
    println!("point_count = {:?}", point_count);
    if point_count < 4 {
        panic!("Point count < 4");
    }
    let max_iterations = (factorial(point_count as u64) / factorial(point_count as u64 - 4)
        * factorial(4 as u64))
    .round() as usize;
    let num_iterations = 500.min(max_iterations);
    let mut already_used_quadruples: HashSet<Vec<usize>> = HashSet::new();
    let mut inlier_lists: Vec<Vec<usize>> = Vec::new();
    for _ in 0..num_iterations {
        let mut random_indices_o: Option<Vec<usize>>;
        loop {
            let _random_indices: Vec<usize> = get_random_quadruple(point_count - 1);
            random_indices_o = Some(_random_indices.clone());
            if !already_used_quadruples.contains(&_random_indices) {
                already_used_quadruples.insert(_random_indices);
                break;
            }
        }
        let random_indices = random_indices_o.unwrap();
        let homography = find_homography(
            random_indices
                .iter()
                .cloned()
                .map(|index| Point {
                    x: matched_key_points[index].0.x,
                    y: matched_key_points[index].0.y,
                })
                .collect(),
            random_indices
                .iter()
                .cloned()
                .map(|index| Point {
                    x: matched_key_points[index].1.x,
                    y: matched_key_points[index].1.y,
                })
                .collect(),
        );
        inlier_lists.push(compute_sift_inliers(&matched_key_points, &homography));
    }
    println!("Inlier lists length = {:?}", inlier_lists.len());
    inlier_lists.sort_unstable_by(|a, b| b.len().cmp(&a.len()));

    let final_inlier_list: Vec<(DescriptedKeyPoint, DescriptedKeyPoint, f64)> = inlier_lists[0]
        .iter()
        .cloned()
        .map(|inliner_index| matched_key_points[inliner_index].clone())
        .collect();

    let final_homography = find_homography(
        final_inlier_list
            .iter()
            .map(|(img_1_key_point, _, _)| Point {
                x: img_1_key_point.x,
                y: img_1_key_point.y,
            })
            .collect(),
        final_inlier_list
            .iter()
            .map(|(_, img_2_key_point, _)| Point {
                x: img_2_key_point.x,
                y: img_2_key_point.y,
            })
            .collect(),
    );
    (final_homography, final_inlier_list)
}

pub fn compute_sift_inliers<'a>(
    matched_key_points: &'a Vec<(DescriptedKeyPoint, DescriptedKeyPoint, f64)>,
    homography: &'a Array2<f64>,
) -> Vec<usize> {
    let mut inlier_indices: Vec<usize> = Vec::new();
    for (i, matched_key_point_pair) in matched_key_points.iter().enumerate() {
        let distance = get_point_distance(
            &project_point_with_homography(
                Point {
                    x: matched_key_point_pair.0.x,
                    y: matched_key_point_pair.0.y,
                },
                homography,
            ),
            &Pointf {
                x: matched_key_point_pair.1.x as f64,
                y: matched_key_point_pair.1.y as f64,
            },
        );
        if distance < SIFT_INLIER_THRESHOLD {
            inlier_indices.push(i);
        }
    }
    inlier_indices
}

pub fn get_point_distance(p1: &Pointf, p2: &Pointf) -> f64 {
    let d_x = p2.x - p1.x;
    let d_y = p2.y - p1.y;
    (d_x * d_x + d_y * d_y).sqrt()
}

pub fn project_point_with_homography<'a>(point: Point, h: &'a Array2<f64>) -> Pointf {
    let Point { x, y } = point;
    let denom = h[[2, 0]] * x as f64 + h[[2, 1]] * y as f64 + h[[2, 2]] + 0.00001;
    Pointf {
        x: (h[[0, 0]] * x as f64 + h[[0, 1]] * y as f64 + h[[0, 2]]) / denom,
        y: (h[[1, 0]] * x as f64 + h[[1, 1]] * y as f64 + h[[1, 2]]) / denom,
    }
}

pub fn get_random_quadruple(max: usize) -> Vec<usize> {
    let mut random_indices_h: HashSet<usize> = HashSet::new();
    while random_indices_h.len() < 4 {
        let random_index = (rand::random::<f64>() * max as f64).floor() as usize;
        if random_indices_h.contains(&random_index) {
            continue;
        }
        random_indices_h.insert(random_index);
    }
    random_indices_h.iter().cloned().collect()
}

pub fn get_homography_from_image_key_points(
    img_1_key_points: &Vec<DescriptedKeyPoint>,
    img_2_key_points: &Vec<DescriptedKeyPoint>,
) -> Array2<f64> {
    println!("-- Matching keypoints --");
    let sorted_key_point_pairs = get_sorted_key_point_pairs(img_1_key_points, img_2_key_points);

    println!("-- Ratio Test Matches --");
    let matched_key_point_pairs =
        get_best_matches_by_ratio_test(img_1_key_points, img_2_key_points, sorted_key_point_pairs);

    let (h, _) = do_ransac(matched_key_point_pairs);

    h
}

fn project_point_in_image(point: Point, descripted_image: &DescriptedImage) -> Option<[f64; 3]> {
    let image = &descripted_image.image;
    let img_shape = image.shape();
    match &descripted_image.homography {
        Some(homography) => {
            let projected_point = project_point_with_homography(point, homography);
            if projected_point.x >= 0.0
                && projected_point.y >= 0.0
                && projected_point.x <= (img_shape[0] - 1) as f64
                && projected_point.y <= (img_shape[1] - 1) as f64
            {
                Some(sample_nd_img(image, projected_point.x, projected_point.y))
            } else {
                None
            }
        }
        None => {
            if point.x >= 0
                && point.y >= 0
                && point.x < (img_shape[0] as i64 - 1)
                && point.y < (img_shape[1] as i64 - 1)
            {
                Some([
                    descripted_image.image[[point.x as usize, point.y as usize, 0]],
                    descripted_image.image[[point.x as usize, point.y as usize, 1]],
                    descripted_image.image[[point.x as usize, point.y as usize, 2]],
                ])
            } else {
                None
            }
        }
    }
}

pub fn do_mean_blend(point: Point, descripted_images: &Vec<&DescriptedImage>) -> [f64; 3] {
    let successful_projections: Vec<[f64; 3]> = descripted_images
        .iter()
        .cloned()
        .map(|descripted_image| project_point_in_image(point, descripted_image))
        .flatten()
        .collect();
    if successful_projections.len() == 0 {
        return [0.0, 0.0, 0.0];
    }
    let blended: Vec<f64> = successful_projections
        .iter()
        .fold([0.0, 0.0, 0.0], |avg_color, color| {
            [
                avg_color[0] + color[0],
                avg_color[1] + color[1],
                avg_color[2] + color[2],
            ]
        })
        .iter()
        .map(|channel| (channel / successful_projections.len() as f64))
        .collect();
    [blended[0], blended[1], blended[2]]
}

pub fn blend_images(descripted_images: Vec<&DescriptedImage>) -> NDRgbImage {
    let all_projected_corners: Vec<Pointf> = descripted_images
        .iter()
        .map(|descripted_image| {
            let shape = descripted_image.image.shape();
            let img_corners = vec![(0, 0), (0, shape[1]), (shape[0], 0), (shape[0], shape[1])];
            match descripted_image.homography.clone() {
                Some(homography) => {
                    let inverse_homography = get_homography_inverse(&homography);
                    img_corners
                        .iter()
                        .map(|(x, y)| {
                            project_point_with_homography(
                                Point {
                                    x: *x as i64,
                                    y: *y as i64,
                                },
                                &inverse_homography,
                            )
                        })
                        .collect::<Vec<Pointf>>()
                }
                None => img_corners
                    .iter()
                    .map(|(x, y)| Pointf {
                        x: *x as f64,
                        y: *y as f64,
                    })
                    .collect::<Vec<Pointf>>(),
            }
        })
        .flatten()
        .collect();
    let (min, max) = {
        let (_min, _max) = all_projected_corners.iter().fold(
            (all_projected_corners[0], all_projected_corners[0]),
            |(min, max), corner| {
                (
                    Pointf {
                        x: if corner.x < min.x { corner.x } else { min.x },
                        y: if corner.y < min.y { corner.y } else { min.y },
                    },
                    Pointf {
                        x: if corner.x > max.x { corner.x } else { max.x },
                        y: if corner.y > max.y { corner.y } else { max.y },
                    },
                )
            },
        );
        (
            Point {
                x: _min.x.floor() as i64,
                y: _min.y.floor() as i64,
            },
            Point {
                x: _max.x.ceil() as i64,
                y: _max.y.ceil() as i64,
            },
        )
    };

    let mut new_image: NDRgbImage =
        Array3::zeros(((max.x - min.x) as usize, (max.y - min.y) as usize, 3));
    println!("New image shape = {:?}", new_image.shape());
    for x in 0..new_image.shape()[0] {
        for y in 0..new_image.shape()[1] {
            let color = do_mean_blend(
                Point {
                    x: min.x + x as i64,
                    y: min.y + y as i64,
                },
                &descripted_images,
            );
            new_image[[x, y, 0]] = color[0];
            new_image[[x, y, 1]] = color[1];
            new_image[[x, y, 2]] = color[2];
        }
    }
    new_image
}
