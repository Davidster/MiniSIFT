use opencv as cv;
use opencv::core::*;
use opencv::imgcodecs::*;
use opencv::prelude::*;
use opencv::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let alpha: i32 = args[2].parse().unwrap();
    let beta: i32 = args[1].parse().unwrap();
    println!("alpha: {}, beta: {}", alpha, beta);
    let mut building_img = cv::imgcodecs::imread(
        "../featureMatcher/image_sets/panorama/pano1_0008.png",
        IMREAD_UNCHANGED,
    )
    .unwrap();
    cv::highgui::imshow("buildingImg", &building_img).unwrap();
    // should be 3: r, g, b
    let imgChannels = building_img.channels().unwrap();
    let imgHeight = building_img.rows();
    let imgWidth = building_img.cols();
    let mut building_img_channels: VectorOfMat = (0..imgChannels)
        .map(|n| {
            Mat::new_rows_cols_with_default(imgHeight, imgWidth, CV_8UC1, Scalar::all(0.0)).unwrap()
        })
        .collect();
    cv::core::split(&building_img, &mut building_img_channels).unwrap();
    let mut building_img_gradient_x_channels: Vector<Mat> = (0..imgChannels)
        .map(|n| {
            Mat::new_rows_cols_with_default(imgHeight, imgWidth, CV_8UC1, Scalar::all(0.0)).unwrap()
        })
        .collect();
    let mut building_img_gradient_y_channels: Vector<Mat> = (0..imgChannels)
        .map(|n| {
            Mat::new_rows_cols_with_default(imgHeight, imgWidth, CV_8UC1, Scalar::all(0.0)).unwrap()
        })
        .collect();
    for (channel_index, channel_name) in (vec![
        String::from("Blue"),
        String::from("Green"),
        String::from("Red"),
    ])
    .iter()
    .enumerate()
    {
        let src = building_img_channels.get(channel_index).unwrap();
        let mut grad_x_out = building_img_gradient_x_channels.get(channel_index).unwrap();
        let mut grad_y_out = building_img_gradient_y_channels.get(channel_index).unwrap();
        cv::imgproc::spatial_gradient(
            &src,
            &mut grad_x_out,
            &mut grad_y_out,
            3,
            cv::core::BorderTypes::BORDER_REPLICATE as i32,
        )
        .unwrap();
        let mut grad_x_out_norm =
            Mat::new_rows_cols_with_default(imgHeight, imgWidth, CV_8UC1, Scalar::all(0.0))
                .unwrap();
        normalize(
            &grad_x_out,
            &mut grad_x_out_norm,
            f64::from(alpha),
            f64::from(beta),
            NORM_MINMAX,
            -1,
            &no_array().unwrap(),
        )
        .unwrap();
        cv::highgui::imshow(
            format!("{} gradient x (norm)", channel_name).as_str(),
            &grad_x_out_norm,
        )
        .unwrap();
        cv::highgui::imshow(format!("{} gradient x", channel_name).as_str(), &grad_x_out).unwrap();
        // cv::highgui::imshow(format!("{} gradient y", channel_name).as_str(), &grad_y_out).unwrap();
    }
    // let mut building_img_gradient_y = Mat::new_rows_cols_with_default(
    //     imgHeight,
    //     imgWidth,
    //     building_img.typ().unwrap(),
    //     Scalar::all(0.0),
    // )
    // .unwrap();

    // cv::highgui::imshow("buildingImgGradientX", &building_img_gradient_x).unwrap();
    // cv::highgui::imshow("buildingImgGradientY", &building_img_gradient_y).unwrap();
    cv::highgui::wait_key(0).unwrap();
    // opencv::imgcodecs::im
    // println!("{:?}", boxesImg);
}

// rainier1Img = cv.imread("image_sets/project_images/Rainier1.png")
// rainier1Gradient = getNpGradient2(rainier1Img)
// rainier1HarrisKeypoints = computeHarrisKeypoints(rainier1Img, rainier1Gradient)
// cv.imwrite("results/1b.png", getDrawKeypointsImg(rainier1Img, rainier1HarrisKeypoints))
