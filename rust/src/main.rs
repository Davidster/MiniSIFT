use image::DynamicImage;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::RgbImage;
use show_image::WindowProxy;
use std::sync::mpsc::channel;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

struct KeyPoint {
    x: u32,
    y: u32,
    radius: f32,
    color: [u8; 3],
}

struct SplitImage {
    r: GrayImage,
    g: GrayImage,
    b: GrayImage,
}

struct ImageGradients {
    x: GrayImage,
    y: GrayImage,
}

fn show_rgb_image(img: RgbImage, name: &str) -> WindowProxy {
    let window = show_image::create_window(name, Default::default()).unwrap();
    window
        .set_image(name, image::DynamicImage::ImageRgb8(img))
        .unwrap();
    window
}

fn show_grayscale_image(img: GrayImage, name: &str) -> WindowProxy {
    let window = show_image::create_window(name, Default::default()).unwrap();
    window
        .set_image(name, image::DynamicImage::ImageLuma8(img))
        .unwrap();
    window
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
    image::open(path).unwrap().as_rgb8().unwrap().to_owned()
}

fn split_img_channels(img: &RgbImage) -> SplitImage {
    let mut r = GrayImage::new(img.width(), img.height());
    let mut g = GrayImage::new(img.width(), img.height());
    let mut b = GrayImage::new(img.width(), img.height());
    for (x, y, color) in img.enumerate_pixels() {
        r.put_pixel(x, y, Luma([color[0]]));
        g.put_pixel(x, y, Luma([color[1]]));
        b.put_pixel(x, y, Luma([color[2]]));
    }
    SplitImage { r, g, b }
}

fn get_image_gradients(img: &GrayImage) -> ImageGradients {
    ImageGradients {
        x: image::imageops::filter3x3(
            img,
            &[1f32, 0f32, -1f32, 2f32, 0f32 - 2f32, 1f32, 0f32, -1f32],
        ),
        y: image::imageops::filter3x3(
            img,
            &[1f32, 2f32, 1f32, 0f32, 0f32, 0f32, -1f32, -2f32, -1f32],
        ),
    }
}

fn draw_keypoints(img: &RgbImage, keypoints: &Vec<KeyPoint>) -> RgbImage {
    let mut result = RgbImage::new(img.width(), img.height());
    for (x, y, _) in img.enumerate_pixels() {
        result.put_pixel(x, y, img.get_pixel(x, y).clone());
    }
    for keypoint in keypoints {
        result.put_pixel(keypoint.x, keypoint.y, Rgb::from(keypoint.color));
    }
    result
}

#[show_image::main]
fn main() {
    let mut windows: Vec<WindowProxy> = Vec::new();

    let building_img =
        load_rgb_image_from_file("../featureMatcher/image_sets/panorama/pano1_0008.png");
    let pixel = building_img.get_pixel(building_img.width() - 1, building_img.height() - 1);
    println!("{}, {}, {}", pixel[0], pixel[1], pixel[2]);

    let building_img_w_keypoints = draw_keypoints(
        &building_img,
        &vec![
            KeyPoint {
                x: 10,
                y: 10,
                radius: 0f32,
                color: [255, 0, 0],
            },
            KeyPoint {
                x: 100,
                y: 10,
                radius: 0f32,
                color: [0, 255, 0],
            },
            KeyPoint {
                x: 100,
                y: 100,
                radius: 0f32,
                color: [0, 0, 255],
            },
            KeyPoint {
                x: 10,
                y: 100,
                radius: 0f32,
                color: [255, 0, 255],
            },
        ],
    );

    let SplitImage {
        g: building_img_g,
        b: building_img_b,
        ..
    } = split_img_channels(&building_img);
    let ImageGradients {
        x: building_img_g_grad_x,
        y: building_img_g_grad_y,
    } = get_image_gradients(&building_img_g);

    windows.push(show_rgb_image(building_img.clone(), "Building"));
    windows.push(show_rgb_image(
        building_img_w_keypoints.clone(),
        "Building w keypoints",
    ));
    windows.push(show_grayscale_image(
        building_img_g.clone(),
        "Building Green",
    ));
    windows.push(show_grayscale_image(
        building_img_b.clone(),
        "Building Blue",
    ));
    windows.push(show_grayscale_image(
        building_img_g_grad_x.clone(),
        "Building Green Grad X",
    ));
    windows.push(show_grayscale_image(
        building_img_g_grad_y.clone(),
        "Building Green Grad Y",
    ));

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
