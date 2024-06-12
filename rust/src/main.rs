mod helpers;
mod image_helpers;
mod sift;


use show_image::WindowProxy;

use std::{env, process};

fn main() {
    show_image::run_context(|| {
        let windows: Vec<WindowProxy> = Vec::new();

        // let window = show_image::create_window("img", Default::default()).unwrap();
        // let width = 500;
        // let height = 500;
        // let mut pixel_map: HashMap<(usize, usize), f64> = HashMap::new();
        // pixel_map.insert((0, 0), 0.0);
        // pixel_map.insert((width, 0), 0.5);
        // pixel_map.insert((0, height), 1.0);
        // pixel_map.insert((width, height), 0.0);
        // // let mut img_nd: ndarray::Array3<f64> = ndarray::Array3::zeros((width, height, 3));
        // let mut img = image::GrayImage::new(width as u32, height as u32);
        // let mut l = 50.0;
        // let mut s = 50.0;
        // // loop {
        // for x in 0..width {
        //     for y in 0..height {
        //         let x2 = width;
        //         let x1 = 0;
        //         let y2 = height;
        //         let y1 = 0;
        //         let interpolated_value = (1.0 / ((x2 - x1) * (y2 - y1)) as f64)
        //             * (pixel_map.get(&(x1, y1)).unwrap() * ((x2 - x) * (y2 - y)) as f64
        //                 + pixel_map.get(&(x2, y1)).unwrap() * ((x - x1) * (y2 - y)) as f64
        //                 + pixel_map.get(&(x1, y2)).unwrap() * ((x2 - x) * (y - y1)) as f64
        //                 + pixel_map.get(&(x2, y2)).unwrap() * ((x - x1) * (y - y1)) as f64);
        //         let as_rgb = colorsys::Rgb::from(colorsys::Hsl::new(
        //             interpolated_value * 360.0,
        //             s % 100.0,
        //             l % 100.0,
        //             None,
        //         ));
        //         img.put_pixel(
        //             x as u32,
        //             y as u32,
        //             image::Luma::from([(interpolated_value * 255.0).round() as u8]),
        //         );
        //     }
        // }
        // // l += 0.1;
        // // s += 0.5;
        // // window
        // //     .set_image("img", image_helpers::ndarray_to_image_rgb(&img_nd))
        // //     .unwrap();
        // window.set_image("img", img.clone()).unwrap();
        // windows.push(window);
        // // thread::sleep(Duration::from_millis(10));
        // // }

        // let kernel = helpers::gaussian_kernel_2d(5, Some(0.85));
        // // println!("{:?}", kernel);

        // let mut min = 100000000000000000.;
        // let mut max = 0.;
        // for i in 0..8 {
        //     for j in 0..8 {
        //         let val = kernel[[i, j]];
        //         if val < min {
        //             min = val;
        //         }
        //         if val > max {
        //             max = val;
        //         }
        //     }
        // }
        // println!("Ratio={:?}", max / min);
        // panic!();

        // let string = String::from("hi");
        // let string_reference = &string;
        // let string_again = *string_reference;

        // Req 1
        // a
        // let boxes_img = image_helpers::load_rgb_image_from_file(
        //     "../featureMatcher/image_sets/project_images/Boxes.png",
        // );
        // let boxes_img_nd = image_helpers::image_to_ndarray_rgb(boxes_img);
        // let boxes_gradients = sift::get_image_gradients(&boxes_img_nd);
        // let boxes_harris_response = sift::compute_harris_response(&boxes_img_nd, &boxes_gradients);
        // windows.push(image_helpers::show_gray_image(
        //     image_helpers::ndarray_to_image_gray(
        //         boxes_gradients.x,
        //         image_helpers::ImgConversionType::NORMALIZE,
        //     ),
        //     "boxes_gradients_x",
        // ));
        // windows.push(image_helpers::show_gray_image(
        //     image_helpers::ndarray_to_image_gray(
        //         boxes_gradients.y,
        //         image_helpers::ImgConversionType::NORMALIZE,
        //     ),
        //     "boxes_gradients_y",
        // ));
        // windows.push(image_helpers::show_gray_image(
        //     image_helpers::ndarray_to_image_gray(
        //         boxes_harris_response,
        //         image_helpers::ImgConversionType::CLAMP,
        //     ),
        //     "boxes_harris_response",
        // ));

        // b
        // let rainier_1_img = image_helpers::load_rgb_image_from_file(
        //     "../featureMatcher/image_sets/project_images/Rainier1.png",
        // );
        // let rainier_1_img_nd = image_helpers::image_to_ndarray_rgb(rainier_1_img.clone());
        // let rainier_1_gradients = sift::get_image_gradients(&rainier_1_img_nd);
        // let rainier_1_harris_key_points =
        //     sift::compute_harris_keypoints(&rainier_1_img_nd, &rainier_1_gradients);
        // windows.push(image_helpers::show_gray_image(
        //     image_helpers::ndarray_to_image_gray(
        //         rainier_1_gradients.x,
        //         image_helpers::ImgConversionType::NORMALIZE,
        //     ),
        //     "rainier_1_gradients_x",
        // ));
        // windows.push(image_helpers::show_gray_image(
        //     image_helpers::ndarray_to_image_gray(
        //         rainier_1_gradients.y,
        //         image_helpers::ImgConversionType::NORMALIZE,
        //     ),
        //     "rainier_1_gradients_y",
        // ));
        // windows.push(image_helpers::show_rgb_image(
        //     image_helpers::draw_key_points(&rainier_1_img, &rainier_1_harris_key_points),
        //     "rainier_1_harris_key_points",
        // ));

        // Req 2
        /* let rainier_1_image = image_helpers::load_rgb_image_from_file(
            // "../featureMatcher/image_sets/project_images/Rainier1.png",
            "./images/left.jpg",
        );
        let rainier_1_image_nd = image_helpers::image_to_ndarray_rgb(&rainier_1_image);
        let rainier_2_image = image_helpers::load_rgb_image_from_file(
            // "../featureMatcher/image_sets/project_images/Rainier2.png",
            "./images/right.jpg",
        );
        let rainier_2_image_nd = image_helpers::image_to_ndarray_rgb(&rainier_2_image);

        let mut rainier_1_descripted_image = sift::DescriptedImage {
            image: rainier_1_image_nd.clone(),
            key_points: sift::do_sift(&rainier_1_image_nd),
            homography: None,
        };

        let mut rainier_2_descripted_image = sift::DescriptedImage {
            image: rainier_2_image_nd.clone(),
            key_points: sift::do_sift(&rainier_2_image_nd),
            homography: None,
        };

        let sorted_key_point_pairs = sift::get_sorted_key_point_pairs(
            &rainier_1_descripted_image.key_points,
            &rainier_2_descripted_image.key_points,
        );
        let ratio_test_matches = sift::get_best_matches_by_ratio_test(
            &rainier_1_descripted_image.key_points,
            &rainier_2_descripted_image.key_points,
            sorted_key_point_pairs,
        ); */

        // let (h, inliers) = sift::do_ransac(ratio_test_matches);
        // let img_1_annotated = image_helpers::draw_key_points(
        //     &rainier_1_image.clone(),
        //     &inliers
        //         .iter()
        //         .map(|(img_1_key_point, _, _)| sift::KeyPoint {
        //             x: img_1_key_point.x,
        //             y: img_1_key_point.y,
        //             value: img_1_key_point.value,
        //             color: img_1_key_point.color,
        //         })
        //         .collect(),
        // );
        // let img_2_annotated = image_helpers::draw_key_points(
        //     &rainier_2_image.clone(),
        //     &inliers
        //         .iter()
        //         .map(|(_, img_2_key_point, _)| sift::KeyPoint {
        //             x: img_2_key_point.x,
        //             y: img_2_key_point.y,
        //             value: img_2_key_point.value,
        //             color: img_2_key_point.color,
        //         })
        //         .collect(),
        // );

        // let img_1_annotated = image_helpers::draw_key_points(
        //     &rainier_1_image.clone(),
        //     &ratio_test_matches
        //         .iter()
        //         .map(|(img_1_key_point, _, _)| sift::KeyPoint {
        //             x: img_1_key_point.x,
        //             y: img_1_key_point.y,
        //             value: img_1_key_point.value,
        //             color: img_1_key_point.color,
        //         })
        //         .collect(),
        // );
        // let img_2_annotated = image_helpers::draw_key_points(
        //     &rainier_2_image.clone(),
        //     &ratio_test_matches
        //         .iter()
        //         .map(|(_, img_2_key_point, _)| sift::KeyPoint {
        //             x: img_2_key_point.x,
        //             y: img_2_key_point.y,
        //             value: img_2_key_point.value,
        //             color: img_2_key_point.color,
        //         })
        //         .collect(),
        // );

        // let img_1_annotated = image_helpers::draw_key_points(
        //     &rainier_1_image.clone(),
        //     &rainier_1_descripted_image
        //         .key_points
        //         .iter()
        //         .map(|img_1_key_point| sift::KeyPoint {
        //             x: img_1_key_point.x,
        //             y: img_1_key_point.y,
        //             value: img_1_key_point.value,
        //             color: img_1_key_point.color,
        //         })
        //         .collect(),
        // );
        // let img_2_annotated = image_helpers::draw_key_points(
        //     &rainier_2_image.clone(),
        //     &rainier_2_descripted_image
        //         .key_points
        //         .iter()
        //         .map(|img_2_key_point| sift::KeyPoint {
        //             x: img_2_key_point.x,
        //             y: img_2_key_point.y,
        //             value: img_2_key_point.value,
        //             color: img_2_key_point.color,
        //         })
        //         .collect(),
        // );

        // windows.push(image_helpers::show_rgb_image(
        //     img_1_annotated.clone(),
        //     "img_1_annotated",
        // ));
        // windows.push(image_helpers::show_rgb_image(
        //     img_2_annotated.clone(),
        //     "img_2_annotated",
        // ));

        /* rainier_2_descripted_image.homography = Some(sift::get_homography_from_image_key_points(
            &rainier_1_descripted_image.key_points,
            &rainier_2_descripted_image.key_points,
        ));

        println!("blendin");

        let blended_image = sift::blend_images(vec![
            &rainier_1_descripted_image,
            &rainier_2_descripted_image,
        ]);

        println!("done blendin");

        windows.push(image_helpers::show_rgb_image(
            rainier_1_image.clone(),
            "rainier_1_image",
        ));
        windows.push(image_helpers::show_rgb_image(
            rainier_2_image.clone(),
            "rainier_2_image",
        ));
        windows.push(image_helpers::show_rgb_image(
            image_helpers::ndarray_to_image_rgb(&blended_image),
            "blended_image",
        )); */

        // let h_nd = find_homography(
        //     vec![
        //         Point { x: 182, y: 313 },
        //         Point { x: 149, y: 327 },
        //         Point { x: 281, y: 489 },
        //         Point { x: 266, y: 468 },
        //     ],
        //     vec![
        //         Point { x: 195, y: 163 },
        //         Point { x: 163, y: 178 },
        //         Point { x: 294, y: 326 },
        //         Point { x: 281, y: 306 },
        //     ],
        // );
        // println!("{:?}", h_nd);

        // let row_1: f64 = *h;

        // println!("{:?}", row_1);

        // println!("{:?}", h.at(1));
        // println!("{:?}", h.at(2));

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
        // let rainier_image =
        //     load_rgb_image_from_file("../featureMatcher/image_sets/project_images/Rainier1.png");
        // let rainier_image_nd = image_to_ndarray_rgb(rainier_image.clone());
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

        // println!("1");
        // let img_gradient = get_image_gradients(&rainier_image_nd);
        // println!("2");
        // let img_harris_keypoints = compute_harris_keypoints(&rainier_image_nd, &img_gradient);
        // println!("3");
        // let rainier_annotated = draw_key_points(&rainier_image, &img_harris_keypoints);
        // let kps: Vec<KeyPoint> = rainier_image_tree_1
        //     .key_points
        //     .unwrap()
        //     .iter()
        //     .cloned()
        //     .map(|key_point| KeyPoint {
        //         x: key_point.x,
        //         y: key_point.y,
        //         value: key_point.value,
        //         color: None,
        //     })
        //     .collect();
        // let rainier_annotated_2 = draw_key_points(&rainier_image, &kps);

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

        if !windows.is_empty() {
            image_helpers::wait_for_windows_to_close(windows);
        }

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
    });
}
