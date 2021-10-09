#![recursion_limit = "256"]

use image::imageops::filter3x3;
use image::DynamicImage;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::RgbImage;

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
    let r = img.slice_move(s![.., .., 0..1]);
    let g = img.slice_move(s![.., .., 1..2]);
    let b = img.slice_move(s![.., .., 2..3]);
    SplitImage { r, g, b }
}

fn merge_img_channels(img: &SplitImage) -> NDRgbImage {
    concatenate(Axis(2), &[img.r.view(), img.g.view(), img.b.view()]).unwrap()
}

fn filter_2d(img: &NDRgbImage, kernel: &Array2<f32>) {
    let img_shape = img.shape();
    let kernel_shape = kernel.shape();
    let half_kernel_width = ((kernel_shape[0] as f32) / 2.).floor() as i32;
    let half_kernel_height = ((kernel_shape[1] as f32) / 2.).floor() as i32;
    let relative_kernel_positions: Array2<i32> =
        Array2<f32>::from(kernel.indexed_iter().map(|((i, j), _)| 0).collect());
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {}
    }
}

fn get_image_gradients(img: &NDRgbImage) -> ImageGradients {
    let sobel_x = [1f32, 0f32, -1f32, 2f32, 0f32 - 2f32, 1f32, 0f32, -1f32];
    let sobel_y = [1f32, 2f32, 1f32, 0f32, 0f32, 0f32, -1f32, -2f32, -1f32];

    let SplitImage { r, g, b } = split_img_channels(img);
    ImageGradients {
        x: merge_img_channels(&SplitImage {
            r: filter3x3(&r, &sobel_x),
            g: filter3x3(&g, &sobel_x),
            b: filter3x3(&b, &sobel_x),
        }),
        y: merge_img_channels(&SplitImage {
            r: filter3x3(&r, &sobel_y),
            g: filter3x3(&g, &sobel_y),
            b: filter3x3(&b, &sobel_y),
        }),
    }
}

fn get_blended_circle_pixel(
    pixel_color: &[u8; 3],
    pixel_x: i32,
    pixel_y: i32,
    circle_color: &[u8; 3],
    circle_center_x: i32,
    circle_center_y: i32,
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
                x as i32,
                y as i32,
                &key_point_colors[i],
                key_point.x as i32,
                key_point.y as i32,
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

    let rainier_image_tree = ImageTreeNode {
        image: load_rgb_image_from_file("../featureMatcher/image_sets/project_images/Rainier1.png"),
        children: Some(vec![ImageTreeNode {
            image: load_rgb_image_from_file("../featureMatcher/image_sets/panorama/pano1_0008.png"),
            children: None,
            key_points: None,
        }]),
        key_points: None,
    };

    let rainier_grad = get_image_gradients(&rainier_image_tree.image);

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

    println!("Opening windows");

    // windows.push(show_rgb_image(building_img.clone(), "Building"));
    windows.push(show_rgb_image(rainier_image_tree.image.clone(), "Rainier"));
    windows.push(show_rgb_image(rainier_grad.x.clone(), "Rainier grad x"));
    windows.push(show_rgb_image(rainier_grad.y.clone(), "Rainier grad y"));

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
