#![recursion_limit = "256"]

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

use std::sync::mpsc::channel;
use std::thread;

use rand::Rng;

type NDRgbImage = Array3<f32>;
type NDGrayImage = Array3<f32>;

struct KeyPoint {
    x: u32,
    y: u32,
    // radius: f64,
    // color: [u8; 3],
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

struct ImageTreeNode {
    image: NDRgbImage,
    key_points: Option<Vec<KeyPoint>>,
    children: Option<Vec<ImageTreeNode>>,
}

fn show_image(img: DynamicImage, name: &str) -> WindowProxy {
    let mut options = WindowOptions::new();
    options = options.set_preserve_aspect_ratio(true);
    // options.
    let window = show_image::create_window(name, options).unwrap();
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
            let clamp = |val: f32| val.min(0f32).max(255f32) as u8;
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

fn ndarray_to_image_grey(img: NDGrayImage, conversion_type: ImgConversionType) -> GrayImage {
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
                ImgConversionType::NORMALIZE => (255f32 * val.min(0f32) / max_val) as u8,
                ImgConversionType::CLAMP => val.min(0f32).max(255f32) as u8,
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
    let half_kernel_width = ((kernel_shape[0] as f32) / 2.).floor() as isize;
    let half_kernel_height = ((kernel_shape[1] as f32) / 2.).floor() as isize;
    let mut relative_kernel_positions = Array3::zeros((kernel_shape[0], kernel_shape[1], 2));
    for x in 0..kernel_shape[0] {
        for y in 0..kernel_shape[1] {
            relative_kernel_positions[[x, y, 0]] = -half_kernel_width + x as isize;
            relative_kernel_positions[[x, y, 1]] = -half_kernel_height + y as isize;
        }
    }
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let mut final_val = 0f32;
            for k_x in 0..kernel_shape[0] {
                for k_y in 0..kernel_shape[1] {
                    let relative_position_x =
                        relative_kernel_positions[[k_x as usize, k_y as usize, 0]] as usize;
                    let relative_position_y =
                        relative_kernel_positions[[k_x as usize, k_y as usize, 1]] as usize;
                    let p_x = (x + relative_position_x).min(0).max(img_shape[0] - 1);
                    let p_y = (y + relative_position_y).min(0).max(img_shape[1] - 1);
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

fn get_image_gradients(img: &NDRgbImage) -> ImageGradients {
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
    ImageGradients {
        x: merge_img_channels(&SplitImage {
            r: filter_2d(&r, &sobel_x),
            g: filter_2d(&g, &sobel_x),
            b: filter_2d(&b, &sobel_x),
        }),
        y: merge_img_channels(&SplitImage {
            r: filter_2d(&r, &sobel_y),
            g: filter_2d(&g, &sobel_y),
            b: filter_2d(&b, &sobel_y),
        }),
    }
}

fn get_blended_circle_pixel(
    pixel_color: &[u8; 3],
    pixel_x: isize,
    pixel_y: isize,
    circle_color: &[u8; 3],
    circle_center_x: isize,
    circle_center_y: isize,
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
                x as isize,
                y as isize,
                &key_point_colors[i],
                key_point.x as isize,
                key_point.y as isize,
                2f64,
                1f32,
            ) {
                result.put_pixel(x, y, Rgb::from(blended_color));
            }
        }
    }
    for (i, key_point) in key_points.iter().enumerate() {
        result.put_pixel(key_point.x, key_point.y, Rgb::from(key_point_colors[i]));
    }
    result
}

// fn compute_harris_response(image: &RgbImage, image_gradients: &ImageGradients) {

// }

// fn compute_harris_keypoints(image: &RgbImage, image_gradients: &ImageGradients) {

// }

fn do_sift(image_tree_node: &mut ImageTreeNode) {
    if image_tree_node.key_points.is_none() {
        println!("-- Harris keypoints --");
        let img_gradient = get_image_gradients(&image_tree_node.image);
        // imgHarrisKeypoints = computeHarrisKeypoints(imgNode["img"], imgGradient)
        // print("-- SIFT Descriptors --")
        // imgKeypoints = getKeypointDescriptors(imgNode["img"], imgGradient, imgHarrisKeypoints)
        // imgNode["keypoints"] = imgKeypoints
    } else {
        println!("Keypoints already computed. Skipping this step");
    }
}

#[show_image::main]
fn main() {
    let mut windows: Vec<WindowProxy> = Vec::new();

    let building_img =
        load_rgb_image_from_file("../featureMatcher/image_sets/panorama/pano1_0008.png");
    let pixel = building_img.get_pixel(building_img.width() - 1, building_img.height() - 1);
    println!("{}, {}, {}", pixel[0], pixel[1], pixel[2]);

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

    // let rainier_image_tree = ImageTreeNode {
    //     image: rainier_image_tree_img,
    //     children: Some(vec![ImageTreeNode {
    //         image: load_rgb_image_from_file("../featureMatcher/image_sets/panorama/pano1_0008.png"),
    //         children: None,
    //         key_points: None,
    //     }]),
    //     key_points: None,
    // };

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

    let rainier_image =
        load_rgb_image_from_file("../featureMatcher/image_sets/project_images/Rainier1.png");
    let rainier_image_nd = image_to_ndarray_rgb(rainier_image.clone());
    let rainier_image_conv = ndarray_to_image_rgb(rainier_image_nd.clone());
    // let rainier_grad = get_image_gradients(&rainier_image_nd);
    // let rainier_grad_img = ndarray_to_image_rgb(get_image_gradients(&rainier_image_nd));

    println!("Opening windows");
    windows.push(show_rgb_image(rainier_image.clone(), "rainier_image"));
    windows.push(show_rgb_image(
        rainier_image_conv.clone(),
        "rainier_image_conv",
    ));
    // windows.push(show_rgb_image(rainier_grad.y.clone(), "Rainier grad y"));

    println!("Windows opened");

    wait_for_windows_to_close(windows);
    // image::
}

// fn wait_for_some(receivers: Vec<std::sync::mpsc::Receiver<()>>) {
//     let (joined_sender, joined_receiver) = std::sync::mpsc::channel();
//     for receiver in receivers {
//         let _tx = joined_sender.clone();
//         thread::spawn(move || {
//             receiver.recv().unwrap();
//             _tx.send(()).unwrap();
//         });
//     }
//     joined_receiver.recv().unwrap();
//     println!("Something was sent on some channel");
// }
