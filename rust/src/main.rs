mod helpers;
mod image_helpers;
mod sift;

use show_image::WindowProxy;

use image_helpers::*;
use sift::*;

#[show_image::main]
fn main() {
    let mut windows: Vec<WindowProxy> = Vec::new();

    // let building_img =
    //     load_rgb_image_from_file("../featureMatcher/image_sets/panorama/pano1_0008.png");
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

    rainier_annotated_2.save("out.png").unwrap();

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
    // windows.push(show_rgb_image(
    //     rainier_annotated.clone(),
    //     "rainier_annotated",
    // ));
    // windows.push(show_rgb_image(
    //     rainier_annotated_2.clone(),
    //     "rainier_annotated_2",
    // ));

    // println!("Windows opened");

    wait_for_windows_to_close(windows);
}
