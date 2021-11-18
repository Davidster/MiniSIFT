mod helpers;
mod image_helpers;
mod sift;

use image;

use std::{env, process};

fn main() {
    let mut cli_args = env::args();

    cli_args.next();

    let image_1_path_string = cli_args
        .next()
        .or_else(|| {
            eprintln!("Missing file path for image 1");
            process::exit(1);
        })
        .unwrap();

    let image_2_path_string = cli_args
        .next()
        .or_else(|| {
            eprintln!("Missing file path for image 2");
            process::exit(1);
        })
        .unwrap();

    let image_1 = image_helpers::load_rgb_image_from_file(&image_1_path_string);
    let image_2 = image_helpers::load_rgb_image_from_file(&image_2_path_string);

    let image_1_nd = image_helpers::image_to_ndarray_rgb(&image_1);
    let image_2_nd = image_helpers::image_to_ndarray_rgb(&image_2);

    let image_1_key_points = sift::do_sift(&image_1_nd);
    let image_2_key_points = sift::do_sift(&image_2_nd);

    let image_1_descripted = sift::DescriptedImage {
        image: image_1_nd,
        key_points: image_1_key_points,
        homography: None,
    };

    let mut image_2_descripted = sift::DescriptedImage {
        image: image_2_nd,
        key_points: image_2_key_points,
        homography: None,
    };

    image_2_descripted.homography = Some(sift::get_homography_from_image_key_points(
        &image_1_descripted.key_points,
        &image_2_descripted.key_points,
    ));

    let blended_image = sift::blend_images(vec![&image_1_descripted, &image_2_descripted]);
    image_helpers::ndarray_to_image_rgb(&blended_image)
        .save_with_format("images/out.jpg", image::ImageFormat::Jpeg)
        .unwrap();
}
