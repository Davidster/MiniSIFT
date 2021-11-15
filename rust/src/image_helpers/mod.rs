use std::sync::mpsc::channel;
use std::thread;

use image::DynamicImage;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::RgbImage;

use ndarray::concatenate;
use ndarray::s;
use ndarray::Array3;
use ndarray::Axis;

use show_image::WindowProxy;

use super::sift::*;

pub type NDRgbImage = Array3<f64>;
pub type NDGrayImage = Array3<f64>;

pub struct SplitImage {
    pub r: NDGrayImage,
    pub g: NDGrayImage,
    pub b: NDGrayImage,
}

#[allow(dead_code)]
pub struct ImageGradients {
    pub x: NDRgbImage,
    pub y: NDRgbImage,
}

pub struct ImageGradientsGray {
    pub x: NDRgbImage,
    pub y: NDRgbImage,
}

#[allow(dead_code)]
pub enum ImgConversionType {
    CLAMP,
    NORMALIZE,
}

pub fn show_image(img: DynamicImage, name: &str) -> WindowProxy {
    let window = show_image::create_window(name, Default::default()).unwrap();
    window.set_image(name, img).unwrap();
    // show_image::back
    window
}

pub fn show_rgb_image(img: RgbImage, name: &str) -> WindowProxy {
    show_image(image::DynamicImage::ImageRgb8(img), name)
}

#[allow(dead_code)]
pub fn show_gray_image(img: GrayImage, name: &str) -> WindowProxy {
    show_image(image::DynamicImage::ImageLuma8(img), name)
}

pub fn wait_for_windows_to_close(windows: Vec<WindowProxy>) {
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

pub fn load_rgb_image_from_file(path: &str) -> RgbImage {
    image::open(path).unwrap().into_rgb8()
}

pub fn split_img_channels(img: &NDRgbImage) -> SplitImage {
    let r = img.clone().slice_move(s![.., .., 0..1]);
    let g = img.clone().slice_move(s![.., .., 1..2]);
    let b = img.clone().slice_move(s![.., .., 2..3]);
    SplitImage { r, g, b }
}

pub fn merge_img_channels(img: &SplitImage) -> NDRgbImage {
    concatenate(Axis(2), &[img.r.view(), img.g.view(), img.b.view()]).unwrap()
}

pub fn rgb_img_to_grayscale(img: &NDRgbImage) -> NDGrayImage {
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

pub fn image_to_ndarray_rgb(img: &RgbImage) -> NDRgbImage {
    let mut out = Array3::zeros((img.width() as usize, img.height() as usize, 3));
    for x in 0..img.width() as usize {
        for y in 0..img.height() as usize {
            let pixel = img.get_pixel(x as u32, y as u32);
            out[[x, y, 0]] = pixel[0] as f64;
            out[[x, y, 1]] = pixel[1] as f64;
            out[[x, y, 2]] = pixel[2] as f64;
        }
    }
    out
}

#[allow(dead_code)]
pub fn image_to_ndarray_gray(img: &GrayImage) -> NDRgbImage {
    let mut out = Array3::zeros((img.width() as usize, img.height() as usize, 0));
    for x in 0..img.width() as usize {
        for y in 0..img.height() as usize {
            let pixel = img.get_pixel(x as u32, y as u32);
            out[[x, y, 0]] = pixel[0] as f64;
        }
    }
    out
}

#[allow(dead_code)]
pub fn ndarray_to_image_rgb(img: &NDRgbImage) -> RgbImage {
    let img_shape = img.shape();
    let mut out = RgbImage::new(img_shape[0] as u32, img_shape[1] as u32);
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let clamp = |val: f64| val.max(0.).min(255.).round() as u8;
            let r = clamp(img[[x, y, 0]]);
            let g = clamp(img[[x, y, 1]]);
            let b = clamp(img[[x, y, 2]]);
            out.put_pixel(x as u32, y as u32, Rgb::from([r, g, b]));
        }
    }
    out
}

#[allow(dead_code)]
pub fn ndarray_to_image_gray(img: &NDGrayImage, conversion_type: ImgConversionType) -> GrayImage {
    let img_shape = img.shape();
    let mut out = GrayImage::new(img_shape[0] as u32, img_shape[1] as u32);
    let mut max_val = 0.;
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
            let convert = |val: f64| match conversion_type {
                ImgConversionType::NORMALIZE => (255. * val.max(0.) / max_val).round() as u8,
                ImgConversionType::CLAMP => val.max(0.).min(255.).round() as u8,
            };
            let r = convert(img[[x, y, 0]]);
            out.put_pixel(x as u32, y as u32, Luma::from([r]));
        }
    }
    out
}

// formula taken from: https://en.wikipedia.org/wiki/Bilinear_interpolation
// TODO: if you give perfect pixels, this returns [NaN, NaN, NaN]
pub fn sample_nd_img(img: &NDRgbImage, x: f64, y: f64) -> [f64; 3] {
    let x1 = x.floor();
    let x2 = x.ceil();
    let y1 = y.floor();
    let y2 = y.ceil();
    let do_interpolation_for_channel = |channel: usize| {
        let corners = [
            img[[x1 as usize, y1 as usize, channel]],
            img[[x2 as usize, y1 as usize, channel]],
            img[[x1 as usize, y2 as usize, channel]],
            img[[x2 as usize, y2 as usize, channel]],
        ];
        (1.0 / ((x2 - x1) * (y2 - y1)))
            * (corners[0] * ((x2 - x) * (y2 - y))
                + corners[1] * ((x - x1) * (y2 - y))
                + corners[2] * ((x2 - x) * (y - y1))
                + corners[3] * ((x - x1) * (y - y1)))
    };
    [
        do_interpolation_for_channel(0),
        do_interpolation_for_channel(1),
        do_interpolation_for_channel(2),
    ]
}

pub fn get_blended_circle_pixel(
    pixel_color: &[u8; 3],
    pixel_x: i64,
    pixel_y: i64,
    circle_color: &[u8; 3],
    circle_center_x: i64,
    circle_center_y: i64,
    circle_radius: f64,
    stroke_size: f64,
) -> Option<[u8; 3]> {
    let dist_x = circle_center_x - pixel_x;
    let dist_y = circle_center_y - pixel_y;
    let dist = ((dist_x.pow(2) + dist_y.pow(2)) as f64).sqrt();
    let alpha = 0f64.max(1. - ((dist - circle_radius).abs() / stroke_size as f64));
    if alpha > 0. {
        let blended_colors: Vec<u8> = circle_color
            .iter()
            .enumerate()
            // interpolate from img_portion to keypoint_portion by alpha
            .map(|(i, channel)| {
                let circle_portion = channel.clone() as f64 * alpha;
                let img_portion = pixel_color[i] as f64 * (1. - alpha);
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

pub fn draw_key_points(img: &RgbImage, key_points: &Vec<KeyPoint>) -> RgbImage {
    let mut result = RgbImage::new(img.width(), img.height());
    let random_key_point_colors: Vec<[u8; 3]> = (0..key_points.len())
        .map(|_| [rand::random(), rand::random(), rand::random()])
        .collect();
    for (x, y, _) in img.enumerate_pixels() {
        let img_color = img.get_pixel(x, y);
        result.put_pixel(x, y, img_color.clone());

        for (i, key_point) in key_points.iter().enumerate() {
            let kp_color = match key_point.color {
                Some(color) => [color[0], color[1], color[2]],
                None => random_key_point_colors[i],
            };
            if let Some(blended_color) = get_blended_circle_pixel(
                &[img_color[0], img_color[1], img_color[2]],
                x as i64,
                y as i64,
                &kp_color,
                key_point.x as i64,
                key_point.y as i64,
                4.,
                1.,
            ) {
                result.put_pixel(x, y, Rgb::from(blended_color));
            }
        }
    }
    for (i, key_point) in key_points.iter().enumerate() {
        let kp_color = match key_point.color {
            Some(color) => [color[0], color[1], color[2]],
            None => random_key_point_colors[i],
        };
        result.put_pixel(key_point.x as u32, key_point.y as u32, Rgb::from(kp_color));
    }
    result
}

pub fn multiply_per_pixel(img1: &NDGrayImage, img2: &NDGrayImage) -> NDGrayImage {
    let img_shape = img1.shape();
    let mut out = Array3::zeros((img_shape[0], img_shape[1], 1));
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            out[[x, y, 0]] = img1[[x, y, 0]] * img2[[x, y, 0]];
        }
    }
    out
}
