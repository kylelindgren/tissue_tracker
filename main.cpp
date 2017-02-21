#include <QApplication>
#include "mc.hpp"
#include "ssim.hpp"
#include "mainwindow.hpp"

#define WRITE    1
#define AUTO_SEL 1
#define SHOW_METRICS 1


int main(void) {
    cv::Mat frame_0_color_L, frame_0_L, frame_comp_L, gradx_comp_L, grady_comp_L;
    cv::Mat frame_0_color_R, frame_0_R, frame_comp_R, gradx_comp_R, grady_comp_R;
    cv::Mat frame_0_color_L_dist, frame_0_color_R_dist;
    cv::Mat calib_mat_L, dist_coef_L, proj_mat_L;
    cv::Mat calib_mat_R, dist_coef_R, proj_mat_R;
    cv::Mat frame_roi;
    cv::Mat MK_L = cv::Mat::zeros(PIXELS, CP_NUM, CV_32FC1);
    cv::Mat MK_R = cv::Mat::zeros(PIXELS, CP_NUM, CV_32FC1);
    cv::Mat J_0_L = cv::Mat::zeros(PIXELS, CP_NUM*2, CV_32FC1);
    cv::Mat J_0_R = cv::Mat::zeros(PIXELS, CP_NUM*2, CV_32FC1);
    cv::Mat X_L = cv::Mat::zeros(PIXELS, 2, CV_16UC1), X_R = cv::Mat::zeros(PIXELS, 2, CV_16UC1);
    cv::Mat roi_proj = cv::Mat(PIXELS, 3, CV_32FC1), roi_3D = cv::Mat(PIXELS, 3, CV_32FC1);

    int n_bins = 256, size_bins = 1, roi_0x_L, roi_0y_L, roi_0x_R, roi_0y_R;
    float correction_L[n_bins], expected_L[n_bins], p_joint_L[n_bins*n_bins];
    float correction_R[n_bins], expected_R[n_bins], p_joint_R[n_bins*n_bins];
//    float mu, mv, tu, tv;
//    float focal;

    cv::Point2i center;
    center.x = -1; center.y = -1;

    std::vector<cv::Point2f> left_features, right_features;

    std::string source_dir = "/home/kylelindgren/cpp_ws/";
    // video sources
    //    VideoCapture cap_L(1); // open the video camera no.
    //    VideoCapture cap_R(2); // open the video camera no.
    //    VideoCapture cap_L("/home/kylelindgren/cpp_ws/mc_src_vids/moving_heart_stereo_L.avi");
    //    VideoCapture cap_R("/home/kylelindgren/cpp_ws/mc_src_vids/moving_heart_stereo_R.avi");
    cv::VideoCapture cap_L(source_dir + "mc_src_vids/moving_heart_stereo_left_depth.avi");
    cv::VideoCapture cap_R(source_dir + "mc_src_vids/moving_heart_stereo_right_depth.avi");
    // output video file
    cv::VideoWriter cap_write(source_dir + "mc_out_vids/mc_stereo_ssim.avi",
                              CV_FOURCC('H', '2', '6', '4'), 100,
                              cv::Size(IMWIDTH*3, IMHEIGHT), false);

    cv::Mat rectif_mat_L, rectif_mat_R;
    cv::Mat E, F, R, T;

    // load camera parameters
    std::string yaml_directory = "/home/kylelindgren/cpp_ws/yamls/";
    calib_mat_L = mc::LoadParameters(yaml_directory +
                                     "left_close_5mm_squares.yaml", "camera_matrix");
    calib_mat_R = mc::LoadParameters(yaml_directory +
                                     "right_close_5mm_squares.yaml", "camera_matrix");
    dist_coef_L = mc::LoadParameters(yaml_directory +
                                     "left_close_5mm_squares.yaml", "distortion_coefficients");
    dist_coef_R = mc::LoadParameters(yaml_directory +
                                     "right_close_5mm_squares.yaml", "distortion_coefficients");
    proj_mat_L  = mc::LoadParameters(yaml_directory +
                                     "left_close_5mm_squares.yaml", "projection_matrix");
    proj_mat_R  = mc::LoadParameters(yaml_directory +
                                     "right_close_5mm_squares.yaml", "projection_matrix");

    calib_mat_L = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "M1");
    calib_mat_R = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "M2");
    dist_coef_L = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "D1");
    dist_coef_R = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "D2");
    E = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "E");
    F = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "F");
    R = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "R");
    T = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "T");

//    std::cout << calib_mat_L << std::endl << calib_mat_R << std::endl << dist_coef_L
//              << std::endl << dist_coef_R << std::endl << E << std::endl << F
//              << std::endl << R << std::endl << T << std::endl;

    cv::Mat rot_rect_L, rot_rect_R, proj_rect_L, proj_rect_R, disp_to_depth;

    cv::stereoRectify(calib_mat_L, dist_coef_L, calib_mat_R, dist_coef_R,
                      cv::Size(IMWIDTH, IMHEIGHT), R, T, rot_rect_L, rot_rect_R,
                      proj_rect_L, proj_rect_R, disp_to_depth);

    cv::Mat cam1map1, cam1map2;
    cv::Mat cam2map1, cam2map2;

    cv::initUndistortRectifyMap(calib_mat_L, dist_coef_L, rot_rect_L, proj_rect_L,
                                cv::Size(IMWIDTH, IMHEIGHT), CV_32FC1, cam1map1, cam1map2);
    cv::initUndistortRectifyMap(calib_mat_R, dist_coef_R, rot_rect_R, proj_rect_R,
                                cv::Size(IMWIDTH, IMHEIGHT), CV_32FC1, cam2map1, cam2map2);

    if (!(cap_L.isOpened() && cap_R.isOpened())) {  // if not success, exit program
        std::cout << "Cannot open the video cameras" << std::endl;
        return -1;
    }
    // take off first few frames to allow auto settings to settle
    bool bSuccess_L, bSuccess_R;
    for (int i = 1; i < 180; i++) {
        bSuccess_L = cap_L.read(frame_0_color_L_dist);
        bSuccess_R = cap_R.read(frame_0_color_R_dist);
        if (!(bSuccess_L && bSuccess_R)) {  // if camera fails to capture, exit
            std::cout << "Cannot read a frame from video streams" << std::endl;
            return -1;
        }
    }

    // compute inverse transpose of left calibration matrix for projection to 3D coord conversion
    cv::Mat C_inv_t = calib_mat_L.clone();
    C_inv_t.convertTo(C_inv_t, CV_32FC1);
    C_inv_t = C_inv_t.inv(CV_LU);
    C_inv_t = C_inv_t.t();
//    std::cout << C_inv_t << std::endl;

    cv::Mat frame_0_L_dist, frame_0_R_dist, frame_L_dist, frame_R_dist;
    // reference images and regions of interest (roi)
    cv::cvtColor(frame_0_color_L_dist, frame_0_L_dist, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame_0_color_R_dist, frame_0_R_dist, cv::COLOR_BGR2GRAY);
//    cv::undistort(frame_0_L_dist, frame_0_L, calib_mat_L, dist_coef_L);
//    cv::undistort(frame_0_R_dist, frame_0_R, calib_mat_R, dist_coef_R);
    cv::remap(frame_0_L_dist, frame_0_L, cam1map1, cam1map2, cv::INTER_LINEAR);
    cv::remap(frame_0_R_dist, frame_0_R, cam2map1, cam2map2, cv::INTER_LINEAR);
    cv::remap(frame_0_color_L_dist, frame_0_color_L, cam1map1, cam1map2, cv::INTER_LINEAR);
    cv::remap(frame_0_color_R_dist, frame_0_color_R, cam2map1, cam2map2, cv::INTER_LINEAR);

    frame_comp_L = frame_0_L.clone();
    frame_comp_R = frame_0_R.clone();

    frame_roi = frame_0_L.clone();
    int num_features = 0;
    cv::Mat frame_L_roi, frame_R_roi, frame_L_dummy, frame_R_dummy;
    frame_0_color_L.copyTo(frame_L_dummy);
    frame_0_color_R.copyTo(frame_R_dummy);
    cv::Rect roi;

    // control point selection
    if (AUTO_SEL) {
        center.x = 215;  // (315,256) for moving_heart, (258,250) for moving_heart_120
        center.y = 221;  // 256
        roi_0x_L = static_cast<int>(center.x-0.5*ROI_W);
        roi_0y_L = static_cast<int>(center.y-0.5*ROI_H);
        roi = cv::Rect(roi_0x_L, roi_0y_L, ROI_W, ROI_H);
        frame_L_roi = frame_L_dummy(roi);
        num_features = mc::MatchFeatures(frame_L_roi, frame_0_color_R,
                                         &left_features, &right_features);
        if (num_features < CP_NUM) {
            std::cout << "Insufficient features! " << "(" << num_features << "/" << CP_NUM << ")"
                      << " Try again." << std::endl << std::endl;
            return -1;
        }
    } else {
        char window_name[24] = "ROI center selection";
        do {
            while (center.x == -1) {
                cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
                cv::imshow(window_name, frame_roi);
                cv::setMouseCallback(window_name, mc::OnMouse, &center);
                cv::waitKey(1);
            }
            cv::destroyWindow(window_name);

            roi_0x_L = static_cast<int>(center.x-0.5*ROI_W);
            roi_0y_L = static_cast<int>(center.y-0.5*ROI_H);

            cv::Rect roi(roi_0x_L, roi_0y_L, ROI_W, ROI_H);
            if (roi_0x_L + ROI_W < frame_L_dummy.cols && roi_0x_L - ROI_W >= 0 && roi_0y_L + ROI_H
                    < frame_L_dummy.rows && roi_0y_L - ROI_H >= 0) {
                frame_L_roi = frame_L_dummy(roi);
                num_features = mc::MatchFeatures(frame_L_roi, frame_0_color_R,
                                                 &left_features, &right_features);
                if (num_features < CP_NUM) {
                    center.x = -1;
                    frame_L_roi.release();
                    left_features.clear(); right_features.clear();
                    std::cout << "Insufficient features! " << "(" << num_features << "/" << CP_NUM
                              << ")" << " Try again." << std::endl << std::endl;
                }
            } else {
                std::cout << "Chosen ROI out of bounds!" << std::endl << std::endl;
                center.x = -1;
            }
        } while (num_features < CP_NUM);
    }

    std::cout << "number of features detected in ROI: " << num_features << std::endl;
    cv::cvtColor(frame_L_roi, frame_L_roi, cv::COLOR_BGR2GRAY);

    // control point storage
    cv::Mat h_0_L(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_L(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_L(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_0_R(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_R(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_R(CP_NUM*2, 1, CV_32FC1);

    cv::Mat h_depth(CP_NUM, 1, CV_32FC1);

    // calc control point depth, h_depth
    double disparity, disparity_tot = 0, depth;
//    mu = calib_mat_L.at<double>(0, 0) / FOCAL;
//    mv = calib_mat_L.at<double>(1, 1) / FOCAL;
//    tu = calib_mat_L.at<double>(0, 2) / mu;
//    tv = calib_mat_L.at<double>(1, 2) / mv;
//    focal = calib_mat_L.at<double>(0, 0);
    for (int i = 0; i < CP_NUM; i++) {
        // using control points from feature matching
        h_0_L.at<float>(i, 0)        = left_features[i].x + roi_0x_L;
        h_0_L.at<float>(i+CP_NUM, 0) = left_features[i].y + roi_0y_L;
        h_0_R.at<float>(i, 0)        = right_features[i].x;
        h_0_R.at<float>(i+CP_NUM, 0) = right_features[i].y;

        disparity = fabs(h_0_L.at<float>(i, 0) - h_0_R.at<float>(i, 0));
//        depth = BASELINE * focal / disparity;
        depth = mc::CalcDepth(disparity);
        h_depth.at<float>(i, 0) = depth;

        // used for estimating roi_0x_L,roi_0y_L for right image
        disparity_tot += fabs((left_features[i].x + roi_0x_L) - (right_features[i].x));
    }
    h_a_L = h_0_L.clone();  // initial control pt values used for 1st current control pt estimates
    h_a_R = h_0_R.clone();

    // fill X matrix with calculated 3D coordinates of all pixels in roi
    uint16_t x_pos, y_pos;
    double dst[4], depth_x, dst_tot, max = 0, min = INFINITY;
    for (int i = 0; i < PIXELS; i++) {
        x_pos = uint16_t(static_cast<int>(i%ROI_W)+roi_0x_L);
        y_pos = uint16_t(static_cast<int>(i/ROI_W)+roi_0y_L);
        X_L.at<ushort>(i, 0) = x_pos;
        X_L.at<ushort>(i, 1) = y_pos;

        dst[0] = sqrt(pow(static_cast<double>(x_pos) -
                          (static_cast<double>(left_features[0].x + roi_0x_L)), 2) +
                        pow(static_cast<double>(y_pos) -
                            (static_cast<double>(left_features[0].y + roi_0y_L)), 2));
        dst[1] = sqrt(pow(static_cast<double>(x_pos) -
                          (static_cast<double>(left_features[1].x + roi_0x_L)), 2) +
                        pow(static_cast<double>(y_pos) -
                            (static_cast<double>(left_features[1].y + roi_0y_L)), 2));
        dst[2] = sqrt(pow(static_cast<double>(x_pos) -
                          (static_cast<double>(left_features[2].x + roi_0x_L)), 2) +
                        pow(static_cast<double>(y_pos) -
                            (static_cast<double>(left_features[2].y + roi_0y_L)), 2));
        dst[3] = sqrt(pow(static_cast<double>(x_pos) -
                          (static_cast<double>(left_features[3].x + roi_0x_L)), 2) +
                        pow(static_cast<double>(y_pos) -
                            (static_cast<double>(left_features[3].y + roi_0y_L)), 2));

        // estimate depth of each pixel from control point depths (inverse distance relationship)
        dst_tot = depth_x = 0;
        for (int i = 0; i < 4; i++) {
            if (dst[i]) {
                dst[i] = 1/dst[i];
                dst_tot += dst[i];
            }
        }
        for (int i = 0; i < CP_NUM; i++)
            depth_x += (dst[i]/dst_tot)*h_depth.at<float>(i, 0);

        if (depth_x > max)  // max and min for displaying roi depth image
            max = depth_x;
        else if (depth_x < min)
            min = depth_x;

        roi_proj.at<float>(i, 2) = depth_x;
    }

    // visual depth image for roi
    /*
    cv::Mat depth_im = cv::Mat::zeros(ROI_H, ROI_W, CV_8UC1);
    for (int r = 0; r < ROI_H; r++)
        for (int c = 0; c < ROI_W; c++)
            depth_im.at<uchar>(r, c) = (roi_proj.at<float>(r*ROI_H+c, 2)-min)*255/(max-min);
    imshow("depth image", depth_im);
    cvWaitKey();
    cvDestroyWindow("depth image");
//*/

    cv::Mat M_L = cv::Mat::zeros(PIXELS, CP_NUM+3, CV_32FC1);
    cv::Mat K_L = cv::Mat::zeros(CP_NUM+3, CP_NUM+3, CV_32FC1);
    cv::Mat M_R = cv::Mat::zeros(PIXELS, CP_NUM+3, CV_32FC1);
    cv::Mat K_R = cv::Mat::zeros(CP_NUM+3, CP_NUM+3, CV_32FC1);

    roi_0x_R = roi_0x_L - static_cast<int>(disparity_tot/CP_NUM);  // rough estimate
    roi_0y_R = roi_0y_L;
    X_R.at<ushort>(0, 0) = roi_0x_R;
    X_R.at<ushort>(0, 1) = roi_0y_R;
    roi = cv::Rect(roi_0x_R, roi_0y_R, ROI_W, ROI_H);
    frame_R_roi = frame_R_dummy(roi);
    cv::cvtColor(frame_R_roi, frame_R_roi, cv::COLOR_BGR2GRAY);
    cv::Mat T_L = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat I_L = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat im_diff_L = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat T_R = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat I_R = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat im_diff_R = cv::Mat::zeros(PIXELS, 1, CV_32FC1);

    // calculate M, get X_R,T
    mc::CalcMXT(h_0_L, h_0_R, frame_0_L, frame_0_R, X_L, &X_R, &M_L, &M_R, &T_L, &T_R);

    // calculate Ks
    mc::CalcK(h_0_L, &K_L);
    mc::CalcK(h_0_R, &K_R);

    cv::Mat K_inv_L = cv::Mat::zeros(CP_NUM+3, CP_NUM+3, CV_32FC1);
    cv::Mat Ks_L = cv::Mat::zeros(CP_NUM+3, CP_NUM, CV_32FC1);
    K_inv_L = K_L.inv(CV_LU);
    Ks_L = K_inv_L(cv::Range(0, CP_NUM+3), cv::Range(0, CP_NUM));
    cv::Mat K_inv_R = cv::Mat::zeros(CP_NUM+3, CP_NUM+3, CV_32FC1);
    cv::Mat Ks_R = cv::Mat::zeros(CP_NUM+3, CP_NUM, CV_32FC1);
    K_inv_R = K_R.inv(CV_LU);
    Ks_R = K_inv_R(cv::Range(0, CP_NUM+3), cv::Range(0, CP_NUM));

    cv::Mat J_i_L   = cv::Mat::zeros(PIXELS, CP_NUM*2, CV_32FC1);
    cv::Mat J_2_L   = cv::Mat::zeros(PIXELS, CP_NUM*2, CV_32FC1);
    cv::Mat J_inv_L = cv::Mat::zeros(CP_NUM*2, PIXELS, CV_32FC1);
    cv::Mat J_i_R   = cv::Mat::zeros(PIXELS, CP_NUM*2, CV_32FC1);
    cv::Mat J_2_R   = cv::Mat::zeros(PIXELS, CP_NUM*2, CV_32FC1);
    cv::Mat J_inv_R = cv::Mat::zeros(CP_NUM*2, PIXELS, CV_32FC1);

    cv::Mat mapx_L = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat mapy_L = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat mapx_R = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat mapy_R = cv::Mat::zeros(PIXELS, 1, CV_32FC1);

    MK_L = M_L*Ks_L;
    MK_R = M_R*Ks_R;

    int iter;
    double delta_h_L;
    double delta_h_R;
    int frame_num = 0;
    double delta = 1e-2;

    cv::Mat frame_L;
    cv::Mat frame_color_L;
    cv::Mat frame_R;
    cv::Mat frame_color_R;
    cv::Mat warped_L;
    cv::Mat warped_R;

    // writing location values on images
    cv::String text;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.75;
    int thickness = 1;
    int baseLine = 0;
    std::list<float> xl, yl, zl;
    float x, y, z;
    x = y = z = 0;
    int qsize = 30;

    // ssim variables
    cv::String text_ssim_L, text_ssim_R;
    float ssim_L, ssim_R;
    double fontScale_ssim = 0.5;

    mc::ResetExpected(n_bins, expected_L, expected_R);

    // image matrices for displaying backwarping and warping
    cv::Mat current_stable_roi(ROI_H, ROI_W*3, CV_8UC1);
    cv::Mat left_roi(current_stable_roi, cv::Rect(0, 0, ROI_W, ROI_H));
    cv::Mat mid_roi(current_stable_roi, cv::Rect(ROI_W, 0, ROI_W, ROI_H));
    cv::Mat right_roi(current_stable_roi, cv::Rect(ROI_W*2, 0, ROI_W, ROI_H));

    cv::Mat current_stable_im(IMHEIGHT, IMWIDTH*3, CV_8UC1);
    cv::Mat left_im(current_stable_im, cv::Rect(0, 0, IMWIDTH, IMHEIGHT));
    cv::Mat mid_im(current_stable_im, cv::Rect(IMWIDTH, 0, IMWIDTH, IMHEIGHT));
    cv::Mat right_im(current_stable_im, cv::Rect(IMWIDTH*2, 0, IMWIDTH, IMHEIGHT));

    double tot_iters = 0, tot_time = 0;

    std::cout << "Program starting..." << std::endl;
    while (1) {
        frame_num++;

        clock_t begin = clock();
        bSuccess_L = cap_L.read(frame_color_L);  // read a new frame from video
        bSuccess_R = cap_R.read(frame_color_R);
        if (!(bSuccess_L && bSuccess_R)) {  // if not success, break loop
            std::cout << "Cannot read a frame from video streams" << std::endl;
            break;
        }
        cv::cvtColor(frame_color_L, frame_L_dist, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame_color_R, frame_R_dist, cv::COLOR_BGR2GRAY);
        cv::remap(frame_L_dist, frame_L, cam1map1, cam1map2, cv::INTER_LINEAR);
        cv::remap(frame_R_dist, frame_R, cam2map1, cam2map2, cv::INTER_LINEAR);

        mc::ComputeExpectedImg(frame_0_L, frame_0_R, correction_L, correction_R,
                           expected_L, expected_R, n_bins, size_bins, &frame_comp_L, &frame_comp_R,
                           &gradx_comp_L, &gradx_comp_R, &grady_comp_L, &grady_comp_R);
        mc::UpdateJ_0(MK_L, MK_R, X_L, X_R, gradx_comp_L, gradx_comp_R,
                  grady_comp_L, grady_comp_R, &J_0_L, &J_0_R);

        // update reference pixel values after lighting compensation
        for (int p = 0; p < PIXELS; p++) {
            T_L.at<float>(p, 0) = frame_comp_L.at<uchar>(X_L.at<ushort>(p, 1),
                                                         X_L.at<ushort>(p, 0));
            T_R.at<float>(p, 0) = frame_comp_R.at<uchar>(X_R.at<ushort>(p, 1),
                                                         X_R.at<ushort>(p, 0));
        }

        dh_L = cv::Scalar(0);
        dh_R = cv::Scalar(0);
        iter = 0;
        do {
            mapx_L = MK_L*h_a_L(cv::Range(0, CP_NUM), cv::Range::all());
            mapy_L = MK_L*h_a_L(cv::Range(CP_NUM, 2*CP_NUM), cv::Range::all());
            mapx_R = MK_R*h_a_R(cv::Range(0, CP_NUM), cv::Range::all());
            mapy_R = MK_R*h_a_R(cv::Range(CP_NUM, 2*CP_NUM), cv::Range::all());
            cv::remap(frame_L, warped_L, mapx_L.reshape(1, ROI_H), mapy_L.reshape(1, ROI_H),
                                                        cv::INTER_LINEAR, 0, cv::Scalar(0));
            cv::remap(frame_R, warped_R, mapx_R.reshape(1, ROI_H), mapy_R.reshape(1, ROI_H),
                                                        cv::INTER_LINEAR, 0, cv::Scalar(0));
            // show warping each iteration
/*
            warped_L.copyTo(left_roi);
            warped_R.copyTo(right_roi);
            frame_L_roi.copyTo(mid_roi);
            cv::imshow("warped left, roi, warped right", current_stable_roi);
            cv::waitKey(1);
//*/

            warped_L.convertTo(warped_L, CV_32FC1);
            warped_R.convertTo(warped_R, CV_32FC1);

            I_L = warped_L.reshape(1, PIXELS);
            I_R = warped_R.reshape(1, PIXELS);

            mc::Jacobian(warped_L, MK_L, &J_i_L);
            mc::Jacobian(warped_R, MK_R, &J_i_R);

            J_2_L = J_i_L + J_0_L;
            J_2_R = J_i_R + J_0_R;
            J_inv_L = (J_2_L.t()*J_2_L).inv(CV_LU)*J_2_L.t();
            J_inv_R = (J_2_R.t()*J_2_R).inv(CV_LU)*J_2_R.t();
            im_diff_L = I_L - T_L;
            im_diff_R = I_R - T_R;

            dh_L = -2*J_inv_L*im_diff_L;
            dh_R = -2*J_inv_R*im_diff_R;
            h_a_L = h_a_L + 2.1*dh_L;
            h_a_R = h_a_R + 2.1*dh_R;

            iter++;

            delta_h_L = 0;
            delta_h_R = 0;
            for (int j = 0; j < CP_NUM*2; j++) {
                delta_h_L += fabs(dh_L.at<float>(j, 0));
                delta_h_R += fabs(dh_R.at<float>(j, 0));
            }
//            std::cout << iter << "\t" << delta_h_L << std::endl;
        } while (delta_h_L > delta && delta_h_R > delta && iter < 30);

        mc::ComputeJointHistogram(n_bins, size_bins, expected_L, expected_R, p_joint_L, p_joint_R,
                                  frame_L, frame_R, frame_0_L, frame_0_R);

        clock_t end = clock();

        // use TPS for depth or just pixel disparity from estimates roi x,y values?
        //        UpdateDepth(focal, h_depth, h_a_L, h_a_R);
        //        roi_proj.col(2) = MK_L*h_depth;

        // calc projective coords (u,v,w) of roi
        for (int i = 0; i < PIXELS; i++)
            roi_proj.at<float>(i, 2) = mc::CalcDepth(mapx_L.at<float>(i, 0) -
                                                     mapx_R.at<float>(i, 0));
//        roi_proj.col(2) = BASELINE * focal / (mapx_L - mapx_R);  // depth in mm -> w = Z
        roi_proj.col(0) = mapx_L.mul(roi_proj.col(2));           // pixel values * depth(w)
        roi_proj.col(1) = mapy_L.mul(roi_proj.col(2));

        roi_3D = roi_proj*C_inv_t;

        text = cv::format("X: %.0fmm, Y: %.0fmm, Z: %.0fmm", x, y, z);

        // using lists for averaging display values
        xl.push_front(roi_3D.at<float>(PIXELS/2, 0));
        yl.push_front(roi_3D.at<float>(PIXELS/2, 1));
        zl.push_front(roi_3D.at<float>(PIXELS/2, 2));
        if (frame_num > qsize) {
            x += xl.front() - xl.back(); y += yl.front() - yl.back(); z += zl.front() - zl.back();
            text = cv::format("X: %.0fmm, Y: %0.0fmm, Z: %.0fmm", x/qsize, y/qsize, z/qsize);
            xl.pop_back(); yl.pop_back(); zl.pop_back();
        } else {
            x += xl.front(); y += yl.front(); z += zl.front();
            text = "";
        }

        cv::Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseLine);
        baseLine += thickness;

        // center the text
        cv::Point textOrg((frame_L.cols - textSize.width)/2, (frame_L.rows - textSize.height));

        frame_L.copyTo(left_im);
        frame_R.copyTo(right_im);
        // illuminate control points for display
        for (int i = 0; i < CP_NUM; i++) {
            left_im.at<uchar>(h_a_L.at<float>(i+CP_NUM, 0), h_a_L.at<float>(i, 0)) = 255;
            right_im.at<uchar>(h_a_R.at<float>(i+CP_NUM, 0), h_a_R.at<float>(i, 0)) = 255;
        }
        mc::DrawROIBorder(MK_L, MK_R, h_a_L, true, &left_im);
        mc::DrawROIBorder(MK_L, MK_R, h_a_R, false, &right_im);
        mc::AffineTrans(MK_L, MK_R, X_L, X_R, h_a_L, true, &left_im);
        mc::AffineTrans(MK_L, MK_R, X_L, X_R, h_a_R, false, &right_im);
        if (SHOW_METRICS) {
            ssim_L = ssim::compute_quality_metrics_L(frame_L_roi, warped_L, 8, 1);
            ssim_R = ssim::compute_quality_metrics_R(frame_R_roi, warped_R, 8, 1);
            text_ssim_L = cv::format("SSIM:%.3f", ssim_L);
            text_ssim_R = cv::format("SSIM:%.3f", ssim_R);
            textSize = getTextSize(text_ssim_L, fontFace, fontScale_ssim, thickness, &baseLine);
            cv::Point text_ssim_Lorg(roi_0x_L + (ROI_W - textSize.width)/2, roi_0y_L-3);
            cv::Point text_ssim_Rorg(roi_0x_R + (ROI_W - textSize.width)/2, roi_0y_R-3);
            putText(left_im,  text_ssim_L, text_ssim_Lorg, fontFace,
                    fontScale_ssim, cv::Scalar::all(255), thickness, 8);
            putText(right_im, text_ssim_R, text_ssim_Rorg, fontFace,
                    fontScale_ssim, cv::Scalar::all(255), thickness, 8);
        }
        mc::DrawInitBorder(roi_0x_L, roi_0y_L, &frame_L);
        putText(frame_L, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        frame_L.copyTo(mid_im);
        cv::imshow("Original stream, Stabilized stream", current_stable_im);
        if (WRITE)
            cap_write.write(current_stable_im);

        std::cout << "exiting frame # " << frame_num << " after " << iter << " iterations and\t"
                  << static_cast<double>(end-begin)/CLOCKS_PER_SEC << " seconds" << std::endl;

        tot_iters += iter;
        tot_time  += static_cast<double>(end-begin)/CLOCKS_PER_SEC;
        if ((cv::waitKey(1) & 0xFF) == 27 || (cv::waitKey(1) & 0xFF) == 'q' || frame_num == 1290) {
            std::cout << "Program ended by user." << std::endl;
            std::cout << "Average iterations: " << tot_iters/frame_num << " after "
                      << frame_num << " frames" << std::endl;
            std::cout << "Average iters time: " << tot_time/frame_num << std::endl;
            break;
        }
    }

    return 0;
}

