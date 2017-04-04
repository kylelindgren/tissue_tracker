#include <QApplication>
#include "mc.hpp"
#include "ssim.hpp"
#include "mainwindow.hpp"

#define WRITE    0
#define AUTO_SEL 1
#define SHOW_METRICS 1


int main(void) {
    cv::Mat frame_0_color_L, frame_0_L, frame_comp_L, gradx_comp_L, grady_comp_L;
    cv::Mat frame_0_color_R, frame_0_R, frame_comp_R, gradx_comp_R, grady_comp_R;
    cv::Mat frame_0_color_L_dist, frame_0_color_R_dist;
    cv::Mat calib_mat_L, dist_coef_L;
    cv::Mat calib_mat_R, dist_coef_R;
    cv::Mat frame_roi;
    cv::Mat MK_L = cv::Mat::zeros(PIXELS, CP_NUM, CV_32FC1);
    cv::Mat MK_R = cv::Mat::zeros(PIXELS, CP_NUM, CV_32FC1);
    cv::Mat J_0_L = cv::Mat::zeros(PIXELS, CP_NUM*2, CV_32FC1);
    cv::Mat J_0_R = cv::Mat::zeros(PIXELS, CP_NUM*2, CV_32FC1);
    cv::Mat X_L = cv::Mat::zeros(PIXELS, 2, CV_16UC1), X_R = cv::Mat::zeros(PIXELS, 2, CV_16UC1);
    cv::Mat roi_proj = cv::Mat(PIXELS, 3, CV_32FC1), roi_3D = cv::Mat(PIXELS, 3, CV_32FC1);

    // camera parameters
    cv::Mat E, F, R, T, rot_rect_L, rot_rect_R, proj_rect_L, proj_rect_R, disp_to_depth;
    cv::Mat cam1map1, cam1map2, cam2map1, cam2map2;

    cv::Mat frame_0_L_dist, frame_0_R_dist, frame_L_dist, frame_R_dist;
    cv::Mat frame_L_roi, frame_R_roi, frame_L_dummy, frame_R_dummy;

    // control point storage
    cv::Mat h_0_L(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_L(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_L_old_conv(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_L(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_0_R(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_R(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_R_old_conv(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_R(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_old_L = cv::Mat::zeros(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_old_R = cv::Mat::zeros(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_sign_L(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_sign_R(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_diff_L = cv::Mat::zeros(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_diff_R = cv::Mat::zeros(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_ddiff_step_L(CP_NUM*2, 1, CV_32FC1);
    cv::Mat dh_ddiff_step_R(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_kal_L = cv::Mat::zeros(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_kal_R = cv::Mat::zeros(CP_NUM*2, 1, CV_32FC1);

    cv::Mat h_depth(CP_NUM, 1, CV_32FC1);

    cv::Mat T_L = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat I_L = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat im_diff_L = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat T_R = cv::Mat::zeros(PIXELS, 1, CV_32FC1);
    cv::Mat I_R = cv::Mat::zeros(PIXELS, 1, CV_32FC1);

    cv::Mat im_diff_R = cv::Mat::zeros(PIXELS, 1, CV_32FC1);

    cv::Mat M_L = cv::Mat::zeros(PIXELS, CP_NUM+3, CV_32FC1);
    cv::Mat K_L = cv::Mat::zeros(CP_NUM+3, CP_NUM+3, CV_32FC1);
    cv::Mat M_R = cv::Mat::zeros(PIXELS, CP_NUM+3, CV_32FC1);
    cv::Mat K_R = cv::Mat::zeros(CP_NUM+3, CP_NUM+3, CV_32FC1);

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
//    cv::VideoCapture cap_L(source_dir + "mc_src_vids/moving_heart_stereo_left_depth.avi");
//    cv::VideoCapture cap_R(source_dir + "mc_src_vids/moving_heart_stereo_right_depth.avi");
//    cv::VideoCapture cap_L(source_dir +
//                     "depth_test_vids/Mar_6/11:32:35_stereo_data/stereo_raw_L_300_x_60fps.avi");
//    cv::VideoCapture cap_R(source_dir +
//                     "depth_test_vids/Mar_6/11:32:35_stereo_data/stereo_raw_R_300_x_60fps.avi");
//    cv::VideoCapture cap_L(source_dir +
//                     "depth_test_vids/Mar_20/10:05:24_stereo_data/stereo_raw_L_300_x_24fps.avi");
//    cv::VideoCapture cap_R(source_dir +
//                     "depth_test_vids/Mar_20/10:05:24_stereo_data/stereo_raw_R_300_x_24fps.avi");
    cv::VideoCapture cap_L(source_dir +
                     "depth_test_vids/Mar_15/13:11:29_stereo_data/stereo_raw_L_300_x_10fps.avi");
    cv::VideoCapture cap_R(source_dir +
                     "depth_test_vids/Mar_15/13:11:29_stereo_data/stereo_raw_R_300_x_10fps.avi");
    int start_frame = 10;
    int end_frame = 540;
    double step_size = 2.0;
    int center_x = 257, center_y = 163;
    // output video file
    cv::VideoWriter cap_write(source_dir + "mc_out_vids/mc_stereo_ssim.avi",
                              CV_FOURCC('H', '2', '6', '4'), 100,
                              cv::Size(IMWIDTH*3, IMHEIGHT), false);
    std::string file_name = source_dir + "/tissue_tracker/cp_loc" + "/a.txt";
    std::ofstream out(file_name.c_str());

    // load camera parameters
    std::string yaml_directory = "/home/kylelindgren/cpp_ws/yamls/";

    calib_mat_L = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "M1");
    calib_mat_R = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "M2");
    dist_coef_L = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "D1");
    dist_coef_R = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "D2");
    E = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "E");
    F = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "F");
    R = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "R");
    T = mc::LoadParameters(yaml_directory + "opencv_calib.yaml", "T");

    cv::stereoRectify(calib_mat_L, dist_coef_L, calib_mat_R, dist_coef_R,
                      cv::Size(IMWIDTH, IMHEIGHT), R, T, rot_rect_L, rot_rect_R,
                      proj_rect_L, proj_rect_R, disp_to_depth);

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
    for (int i = 1; i < start_frame; i++) {
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

    // reference images and regions of interest (roi)
    cv::cvtColor(frame_0_color_L_dist, frame_0_L_dist, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame_0_color_R_dist, frame_0_R_dist, cv::COLOR_BGR2GRAY);
    cv::remap(frame_0_L_dist, frame_0_L, cam1map1, cam1map2, cv::INTER_LINEAR);
    cv::remap(frame_0_R_dist, frame_0_R, cam2map1, cam2map2, cv::INTER_LINEAR);
    cv::remap(frame_0_color_L_dist, frame_0_color_L, cam1map1, cam1map2, cv::INTER_LINEAR);
    cv::remap(frame_0_color_R_dist, frame_0_color_R, cam2map1, cam2map2, cv::INTER_LINEAR);

    frame_comp_L = frame_0_L.clone();
    frame_comp_R = frame_0_R.clone();

    frame_roi = frame_0_L.clone();
    int num_features = 0;
    frame_0_color_L.copyTo(frame_L_dummy);
    frame_0_color_R.copyTo(frame_R_dummy);
    cv::Rect roi;

    // control point selection
    if (AUTO_SEL) {
        center.x = center_x;
        center.y = center_y;
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

    // calc control point depth, h_depth
    double disparity, disparity_tot = 0, depth;
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
    for (int i = 0; i < PIXELS; i++) {
        x_pos = uint16_t(static_cast<int>(i%ROI_W)+roi_0x_L);
        y_pos = uint16_t(static_cast<int>(i/ROI_W)+roi_0y_L);
        X_L.at<ushort>(i, 0) = x_pos;
        X_L.at<ushort>(i, 1) = y_pos;
    }

    roi_0x_R = roi_0x_L - static_cast<int>(disparity_tot/CP_NUM);  // rough estimate
    roi_0y_R = roi_0y_L;
    X_R.at<ushort>(0, 0) = roi_0x_R;
    X_R.at<ushort>(0, 1) = roi_0y_R;
    roi = cv::Rect(roi_0x_R, roi_0y_R, ROI_W, ROI_H);
    frame_R_roi = frame_R_dummy(roi);
    cv::cvtColor(frame_R_roi, frame_R_roi, cv::COLOR_BGR2GRAY);

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
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.75;
    int font_thickness = 1;
    int font_baseline = 0;
    std::list<float> xl, yl, zl;
    std::list<float> cp_steps_L[CP_NUM*2];
    std::list<float> cp_steps_R[CP_NUM*2];
    float x, y, z;
    x = y = z = 0;
    int qsize = 30;
    int cp_steps = 2;

    // ssim variables
    cv::String text_ssim_L, text_ssim_R;
    float ssim_L, ssim_R;
    double font_scale_ssim = 0.5;

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
    cv::Mat dh_L_tot = cv::Mat::zeros(2*CP_NUM, 1, CV_32FC1);
    cv::Mat dh_R_tot = cv::Mat::zeros(2*CP_NUM, 1, CV_32FC1);
    cv::Mat h_a_L_old = cv::Mat::zeros(CP_NUM*2, 1, CV_32FC1);
    cv::Mat h_a_R_old = cv::Mat::zeros(CP_NUM*2, 1, CV_32FC1);
    h_a_L_old = h_0_L.clone();
    h_a_R_old = h_0_R.clone();

    // hannaford controller
//    h_0_L = cv::Scalar(0);  // for testing against known functions
//    h_0_R = cv::Scalar(0);
    float lambda = 15.0, beta = 0.001;
    float max_D_L = 0, max_D_R = 0, cp_step_size = 1;
    cv::Mat X_hat_L = cv::Mat::zeros(2*CP_NUM, 1, CV_32FC1);
    cv::Mat X_hat_R = cv::Mat::zeros(2*CP_NUM, 1, CV_32FC1);
    cv::Mat e_L = cv::Mat::zeros(2*CP_NUM, 1, CV_32FC1);
    cv::Mat e_R = cv::Mat::zeros(2*CP_NUM, 1, CV_32FC1);
    cv::Mat G_L = cv::Mat::zeros(3, 2*CP_NUM, CV_32FC1);
    cv::Mat G_R = cv::Mat::zeros(3, 2*CP_NUM, CV_32FC1);
    cv::Mat G_L_old = cv::Mat::zeros(3, 2*CP_NUM, CV_32FC1);
    cv::Mat G_R_old = cv::Mat::zeros(3, 2*CP_NUM, CV_32FC1);
    cv::Mat A_L = cv::Mat::zeros(3, 2*CP_NUM, CV_32FC1);
    cv::Mat A_R = cv::Mat::zeros(3, 2*CP_NUM, CV_32FC1);
    cv::Mat D_mat_L = cv::Mat::zeros(3, 2*CP_NUM, CV_32FC1);
    cv::Mat D_mat_R = cv::Mat::zeros(3, 2*CP_NUM, CV_32FC1);
    cv::Mat e_cp_L = cv::Mat::zeros(2*CP_NUM, 1, CV_32FC1);
    cv::Mat e_cp_R = cv::Mat::zeros(2*CP_NUM, 1, CV_32FC1);
    std::list<float> D_L[CP_NUM*2];
    std::list<float> D_R[CP_NUM*2];
    // for cp locations
    h_0_L.copyTo(X_hat_L);
    h_0_R.copyTo(X_hat_R);
    // for cp step size
    X_hat_L = cv::Scalar(step_size);
    X_hat_R = cv::Scalar(step_size);
    // initialize data vector with initial control point locations
    /*
    for (int i = 0; i < 3; i++) {
        D_L[0].push_front(h_0_L.at<float>(0, 0));
        D_L[1].push_front(h_0_L.at<float>(1, 0));
        D_L[2].push_front(h_0_L.at<float>(2, 0));
        D_L[3].push_front(h_0_L.at<float>(3, 0));
        D_L[4].push_front(h_0_L.at<float>(4, 0));
        D_L[5].push_front(h_0_L.at<float>(5, 0));
        D_L[6].push_front(h_0_L.at<float>(6, 0));
        D_L[7].push_front(h_0_L.at<float>(7, 0));

        D_R[0].push_front(h_0_R.at<float>(0, 0));
        D_R[1].push_front(h_0_R.at<float>(1, 0));
        D_R[2].push_front(h_0_R.at<float>(2, 0));
        D_R[3].push_front(h_0_R.at<float>(3, 0));
        D_R[4].push_front(h_0_R.at<float>(4, 0));
        D_R[5].push_front(h_0_R.at<float>(5, 0));
        D_R[6].push_front(h_0_R.at<float>(6, 0));
        D_R[7].push_front(h_0_R.at<float>(7, 0));
    }
    */
    // initialize data vector with initial step sizes
    for (int i = 0; i < 3; i++) {
        D_L[0].push_front(cp_step_size);
        D_L[1].push_front(cp_step_size);
        D_L[2].push_front(cp_step_size);
        D_L[3].push_front(cp_step_size);
        D_L[4].push_front(cp_step_size);
        D_L[5].push_front(cp_step_size);
        D_L[6].push_front(cp_step_size);
        D_L[7].push_front(cp_step_size);

        D_R[0].push_front(cp_step_size);
        D_R[1].push_front(cp_step_size);
        D_R[2].push_front(cp_step_size);
        D_R[3].push_front(cp_step_size);
        D_R[4].push_front(cp_step_size);
        D_R[5].push_front(cp_step_size);
        D_R[6].push_front(cp_step_size);
        D_R[7].push_front(cp_step_size);
    }
    // convert D from list to mat
    for (int j = 0; j < CP_NUM; j++) {
        int i = 0;
        for (auto &p : D_L[j]) {
            D_mat_L.at<float>(i++, j) = p;
        }
        i = 0;
        for (auto &p : D_R[j]) {
            D_mat_R.at<float>(i++, j) = p;
        }
    }

    for (int i = 0; i < 2*CP_NUM; i++) {
        A_L.at<float>(0, i) = 1.0;
        A_L.at<float>(1, i) = -0.0;
        A_R.at<float>(0, i) = 1.0;
        A_R.at<float>(1, i) = -0.0;
    }

    // for cp convergence step recording
    h_0_L.copyTo(h_a_L_old_conv);
    h_0_R.copyTo(h_a_R_old_conv);

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
        e_cp_L = cv::Scalar(0);
        e_cp_R = cv::Scalar(0);
        delta_h_L = delta_h_R = 10;
        iter = 0;
        do {
            // run left and right together, exit when both converge
/*
            mapx_L = MK_L*h_a_L(cv::Range(0, CP_NUM), cv::Range::all());
            mapy_L = MK_L*h_a_L(cv::Range(CP_NUM, 2*CP_NUM), cv::Range::all());
            mapx_R = MK_R*h_a_R(cv::Range(0, CP_NUM), cv::Range::all());
            mapy_R = MK_R*h_a_R(cv::Range(CP_NUM, 2*CP_NUM), cv::Range::all());
            cv::remap(frame_L, warped_L, mapx_L.reshape(1, ROI_H), mapy_L.reshape(1, ROI_H),
                                                        cv::INTER_LINEAR, 0, cv::Scalar(0));
            cv::remap(frame_R, warped_R, mapx_R.reshape(1, ROI_H), mapy_R.reshape(1, ROI_H),
                                                        cv::INTER_LINEAR, 0, cv::Scalar(0));
            // show warping each iteration

            warped_L.copyTo(left_roi);
            warped_R.copyTo(right_roi);
            frame_L_roi.copyTo(mid_roi);
            cv::imshow("warped left, roi, warped right", current_stable_roi);
            cv::waitKey(1);


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
            h_a_L += X_hat_L.mul(dh_L);  // * (exp(-(static_cast<double>(iter/2))) + 1);
            h_a_R += X_hat_R.mul(dh_R);  // * (exp(-(static_cast<double>(iter/2))) + 1);
            // find dh switchbacks
            dh_sign_L = dh_L.mul(dh_old_L);
            dh_sign_R = dh_R.mul(dh_old_R);
            for (int i = 0; i < CP_NUM*2; i++) {
                if (dh_sign_L.at<float>(i, 0) < 0)
                    e_cp_L.at<float>(i, 0)++;
                if (dh_sign_R.at<float>(i, 0) < 0)
                    e_cp_R.at<float>(i, 0)++;
            }
            dh_L.copyTo(dh_old_L);
            dh_R.copyTo(dh_old_R);

//            */

            ///
            /// try splitting up left and right iterations
            ///
///*
            if (delta_h_L > delta) {
                mapx_L = MK_L*h_a_L(cv::Range(0, CP_NUM), cv::Range::all());
                mapy_L = MK_L*h_a_L(cv::Range(CP_NUM, 2*CP_NUM), cv::Range::all());
                cv::remap(frame_L, warped_L, mapx_L.reshape(1, ROI_H), mapy_L.reshape(1, ROI_H),
                                                            cv::INTER_LINEAR, 0, cv::Scalar(0));

                warped_L.convertTo(warped_L, CV_32FC1);

                I_L = warped_L.reshape(1, PIXELS);

                mc::Jacobian(warped_L, MK_L, &J_i_L);

                J_2_L = J_i_L + J_0_L;
                J_inv_L = (J_2_L.t()*J_2_L).inv(CV_LU)*J_2_L.t();
                im_diff_L = I_L - T_L;

                dh_L = -2*J_inv_L*im_diff_L;
//                if (iter < 3)
                    h_a_L += X_hat_L.mul(dh_L);  // * (exp(-(static_cast<double>(iter/2))) + 1);
//                else
//                    h_a_L += dh_L;
                // find dh switchbacks
//                dh_sign_L = dh_L.mul(dh_old_L);
//                for (int i = 0; i < CP_NUM*2; i++) {
//                    if (dh_sign_L.at<float>(i, 0) < 0)
//                        e_cp_L.at<float>(i, 0)++;
//                }
//                dh_L.copyTo(dh_old_L);

                // record steps toward convergence
//                out << h_a_L.at<float>(0, 0) - h_a_L_old_conv.at<float>(0, 0) << " ";
//                h_a_L.copyTo(h_a_L_old_conv);
            }

            if (delta_h_R > delta) {
                mapx_R = MK_R*h_a_R(cv::Range(0, CP_NUM), cv::Range::all());
                mapy_R = MK_R*h_a_R(cv::Range(CP_NUM, 2*CP_NUM), cv::Range::all());
                cv::remap(frame_R, warped_R, mapx_R.reshape(1, ROI_H), mapy_R.reshape(1, ROI_H),
                                                            cv::INTER_LINEAR, 0, cv::Scalar(0));

                warped_R.convertTo(warped_R, CV_32FC1);

                I_R = warped_R.reshape(1, PIXELS);

                mc::Jacobian(warped_R, MK_R, &J_i_R);

                J_2_R = J_i_R + J_0_R;
                J_inv_R = (J_2_R.t()*J_2_R).inv(CV_LU)*J_2_R.t();
                im_diff_R = I_R - T_R;

                dh_R = -2*J_inv_R*im_diff_R;
//                if (iter < 3)
                    h_a_R += X_hat_R.mul(dh_R);  // * (exp(-(static_cast<double>(iter/2))) + 1);
//                else
//                    h_a_R += dh_R;
                // find dh switchbacks
//                dh_sign_R = dh_R.mul(dh_old_R);
//                for (int i = 0; i < CP_NUM*2; i++) {
//                    if (dh_sign_R.at<float>(i, 0) < 0)
//                        e_cp_R.at<float>(i, 0)++;
//                }
//                dh_R.copyTo(dh_old_R);
            }
//*/

            iter++;

            delta_h_L = delta_h_R = 0;
            for (int j = 0; j < CP_NUM*2; j++) {
                delta_h_L += fabs(dh_L.at<float>(j, 0));
                delta_h_R += fabs(dh_R.at<float>(j, 0));
            }

        } while (delta_h_L > delta && delta_h_R > delta && iter < 30);

//        out << "\n";

        // testing hannaford controller with known functions
//        for (int i = 0; i < 2*CP_NUM; i++) {
            // lambda = 15, beta = 0.7, A += beta*G, A0 = 1, A1 = 0, => w/o G normalizing
            // lambda = 15, beta = <1, A += beta*G, A0 = 1, A1 = 0, => with G normalizing
//            h_a_L.at<float>(i, 0) = 1*sin(static_cast<float>(frame_num)/50);
//            h_a_R.at<float>(i, 0) = 1*sin(static_cast<float>(frame_num)/50);
            // lambda = 15, beta = 0.001, A += beta*G, A0 = 1, A1 = 0, => with G normalizing
//            h_a_L.at<float>(i, 0) = frame_num*1;
//            h_a_R.at<float>(i, 0) = frame_num*1;
//        }
//        std::cout << "h_a_L(0, 0): " << h_a_L.at<float>(0, 0) << std::endl;
        ///
        /// controller updating cp point locations
        ///
/*
        // compute error
        e_L = h_a_L - X_hat_L;
        e_R = h_a_R - X_hat_R;
        std::cout << "e_L(0, 0):   " << e_L.at<float>(0, 0) << std::endl;
        std::cout << "D_L:\n" << D_mat_L.col(0) << std::endl;
        // estimate error function gradient
        for (int i = 0; i < 2*CP_NUM; i++) {
            max_D_L = fabs(D_mat_L.at<float>(0, i));
            max_D_R = fabs(D_mat_R.at<float>(0, i));
            if (!max_D_L) max_D_L = 1;
            if (!max_D_R) max_D_R = 1;
            for (int j = 1; j < 3; j++) {
                if (fabs(D_mat_L.at<float>(j, i)) > max_D_L)
                    max_D_L = fabs(D_mat_L.at<float>(j, i));
                if (fabs(D_mat_R.at<float>(j, i)) > max_D_R)
                    max_D_R = fabs(D_mat_R.at<float>(j, i));
            }
            G_L.col(i) = exp(-1/lambda)*G_L_old.col(i) +
                    e_L.at<float>(i, 0)*(D_mat_L.col(i)/lambda)/max_D_L;
            G_R.col(i) = exp(-1/lambda)*G_R_old.col(i) +
                    e_R.at<float>(i, 0)*(D_mat_R.col(i)/lambda)/max_D_R;
        }
//        std::cout << exp(-1/lambda)*G_L_old.col(0) << std::endl;
//        std::cout << e_L.at<float>(0, 0)*D_mat_L.col(0)/lambda << std::endl;
        G_L.copyTo(G_L_old);
        G_R.copyTo(G_R_old);
        // adapt prediction coefficients
        for (int i = 0; i < 2*CP_NUM; i++) {
            A_L.col(i) += beta*G_L.col(i);
            A_R.col(i) += beta*G_R.col(i);
        }
        std::cout << "A_L col 0:\n" << A_L.col(0) << std::endl;
        std::cout << "G_L col 0:\n" << G_L.col(0) << std::endl;
        // update prediction data
        for (int i = 0; i < 2*CP_NUM; i++) {
            D_L[i].push_front(h_a_L.at<float>(i, 0));
            D_L[i].pop_back();
            D_R[i].push_front(h_a_R.at<float>(i, 0));
            D_R[i].pop_back();
        } // convert D from list to mat
        for (int j = 0; j < 2*CP_NUM; j++) {
            int i = 0;
            for (auto &p : D_L[j]) {
                D_mat_L.at<float>(i++, j) = p;
            }
            i = 0;
            for (auto &p : D_R[j]) {
                D_mat_R.at<float>(i++, j) = p;
            }
        }
        // compute new predictor
        for (int i = 0; i < 2*CP_NUM; i++) {
            X_hat_L.row(i) = A_L.col(i).t()*D_mat_L.col(i);
            X_hat_R.row(i) = A_R.col(i).t()*D_mat_R.col(i);
        }
        // update estimate
        X_hat_L.copyTo(h_a_L);
        X_hat_R.copyTo(h_a_R);
        std::cout << std::endl;
//*/

        ///
        /// controller updating cp step size
        ///
/*
        // compute error
        // e_cp_L/R computed in iter loop

//        std::cout << "e_cp_L.col(0):\n" << e_cp_L.col(0) << std::endl;
//        std::cout << "X_hat_L.col(0):\n" << X_hat_L.col(0) << std::endl;

//        std::cout << "D_L:\n" << D_mat_L.col(0) << std::endl;
        // estimate error function gradient
//        for (int i = 0; i < 2*CP_NUM; i++) {
//            max_D_L = fabs(D_mat_L.at<float>(0, i));
//            max_D_R = fabs(D_mat_R.at<float>(0, i));
//            if (!max_D_L) max_D_L = 1;
//            if (!max_D_R) max_D_R = 1;
//            for (int j = 1; j < 3; j++) {
//                if (fabs(D_mat_L.at<float>(j, i)) > max_D_L)
//                    max_D_L = fabs(D_mat_L.at<float>(j, i));
//                if (fabs(D_mat_R.at<float>(j, i)) > max_D_R)
//                    max_D_R = fabs(D_mat_R.at<float>(j, i));
//            }
//            G_L.col(i) = exp(-1/lambda)*G_L_old.col(i) +
//                    e_cp_L.at<float>(i, 0)*(D_mat_L.col(i)/lambda)/max_D_L;
//            G_R.col(i) = exp(-1/lambda)*G_R_old.col(i) +
//                    e_cp_R.at<float>(i, 0)*(D_mat_R.col(i)/lambda)/max_D_R;
//        }
////        std::cout << exp(-1/lambda)*G_L_old.col(0) << std::endl;
////        std::cout << e_L.at<float>(0, 0)*D_mat_L.col(0)/lambda << std::endl;
//        G_L.copyTo(G_L_old);
//        G_R.copyTo(G_R_old);
//        // adapt prediction coefficients
//        for (int i = 0; i < 2*CP_NUM; i++) {
//            A_L.col(i) += beta*G_L.col(i);
//            A_R.col(i) += beta*G_R.col(i);
//        }
////        std::cout << "A_L col 0:\n" << A_L.col(0) << std::endl;
////        std::cout << "G_L col 0:\n" << G_L.col(0) << std::endl;
////         update prediction data
//        for (int i = 0; i < 2*CP_NUM; i++) {
//            D_L[i].push_front(X_hat_L.at<float>(i, 0));
//            D_L[i].pop_back();
//            D_R[i].push_front(X_hat_R.at<float>(i, 0));
//            D_R[i].pop_back();
//        } // convert D from list to mat
//        for (int j = 0; j < 2*CP_NUM; j++) {
//            int i = 0;
//            for (auto &p : D_L[j]) {
//                D_mat_L.at<float>(i++, j) = p;
//            }
//            i = 0;
//            for (auto &p : D_R[j]) {
//                D_mat_R.at<float>(i++, j) = p;
//            }
//        }
        // compute new predictor
//        for (int i = 0; i < 2*CP_NUM; i++) {
////            X_hat_L.row(i) = A_L.col(i).t()*D_mat_L.col(i);
////            X_hat_R.row(i) = A_R.col(i).t()*D_mat_R.col(i);
//            X_hat_L.at<float>(i, 0) -= (e_cp_L.at<float>(i, 0) - 1) / 30;
//            X_hat_R.at<float>(i, 0) -= (e_cp_R.at<float>(i, 0) - 1) / 30;
//        }
//        std::cout << std::endl;
//*/

        //// control updating

//        for (int i = 0; i < 2*CP_NUM; i++) {
//            cp_steps_L[i].push_front(h_a_L.at<float>(i, 0)
//                                     - h_a_L_old.at<float>(i, 0));
//            cp_steps_R[i].push_front(h_a_R.at<float>(i, 0)
//                                     - h_a_R_old.at<float>(i, 0));
//        }
//        if (frame_num > cp_steps) {
//            for (int i = 0; i < 2*CP_NUM; i++) {
//                cp_steps_L[i].pop_back();
//                cp_steps_R[i].pop_back();
//            }
//            for (int i = 0; i < 2*CP_NUM; i++) {
//                h_a_L.at<float>(i, 0) += std::accumulate(std::begin(cp_steps_L[i]),
//                                                         std::end(cp_steps_L[i]), 0.0) / cp_steps;
//                h_a_R.at<float>(i, 0) += std::accumulate(std::begin(cp_steps_R[i]),
//                                                         std::end(cp_steps_R[i]), 0.0) / cp_steps;
//            }
//        }

////        for (int i = 0; i < 2*CP_NUM; i++) {
////            h_a_L.at<float>(i, 0) += std::accumulate(std::begin(cp_steps_L[i]),
////                                                     std::end(cp_steps_L[i]), 0.0) / cp_steps;
////            h_a_R.at<float>(i, 0) += std::accumulate(std::begin(cp_steps_R[i]),
////                                                     std::end(cp_steps_R[i]), 0.0) / cp_steps;
////        }


        dh_ddiff_step_L = 2*(h_a_L - h_a_L_old) - dh_diff_L;  // 2*(new loc diff) - old loc diff
        dh_ddiff_step_R = 2*(h_a_R - h_a_R_old) - dh_diff_R;

        dh_diff_L = h_a_L - h_a_L_old;  // new loc diff
        dh_diff_R = h_a_R - h_a_R_old;

        // kalman error for step
//        for (int j = 0; j < 2*CP_NUM; j++)
//            out << (h_a_L.at<float>(j, 0) - h_a_L_old.at<float>(j, 0))
//                   - h_a_kal_L.at<float>(j, 0) << " ";
//        for (int j = 0; j < 2*CP_NUM; j++)
//            out << (h_a_R.at<float>(j, 0) - h_a_R_old.at<float>(j, 0))
//                   - h_a_kal_R.at<float>(j, 0) << " ";

        h_a_L.copyTo(h_a_L_old);
        h_a_R.copyTo(h_a_R_old);

        // kalman error for loc
//        for (int j = 0; j < 2*CP_NUM; j++)
//            out << h_a_L.at<float>(j, 0) - h_a_kal_L.at<float>(j, 0) << " ";
//        for (int j = 0; j < 2*CP_NUM; j++)
//            out << h_a_R.at<float>(j, 0) - h_a_kal_R.at<float>(j, 0) << " ";


        // control point pixel differences between frames (steps)
//        if (frame_num == 1) {  // first
//            for (int j = 0; j < 2*CP_NUM; j++)
//                out << h_a_L.at<float>(j, 0) - h_0_L.at<float>(j, 0) << " ";
//            for (int j = 0; j < 2*CP_NUM; j++)
//                out << h_a_R.at<float>(j, 0) - h_0_R.at<float>(j, 0) << " ";
//            for (int j = 0; j < 2*CP_NUM; j++)
//                out << h_a_L.at<float>(j, 0) - h_0_L.at<float>(j, 0) << " ";
//            for (int j = 0; j < 2*CP_NUM; j++)
//                out << h_a_R.at<float>(j, 0) - h_0_R.at<float>(j, 0) << " ";
//            out << "\n";
//        }
//        for (int j = 0; j < 2*CP_NUM; j++)
//            out << dh_diff_L.at<float>(j, 0) << " ";
//        for (int j = 0; j < 2*CP_NUM; j++)
//            out << dh_diff_R.at<float>(j, 0) << " ";
//        // Kalman filter estimate for next cp iter STEP
//        mc::KalmanStepCP(&dh_ddiff_step_L, &dh_ddiff_step_R, 5.0, 12.0);
//        dh_ddiff_step_L.copyTo(h_a_kal_L);
//        dh_ddiff_step_R.copyTo(h_a_kal_R);
//        for (int j = 0; j < 2*CP_NUM; j++)
//            out << h_a_kal_L.at<float>(j, 0) << " ";
//        for (int j = 0; j < 2*CP_NUM; j++)
//            out << h_a_kal_R.at<float>(j, 0) << " ";
//        h_a_L += dh_ddiff_step_L;
//        h_a_R += dh_ddiff_step_R;

        // control point locations
        if (frame_num == 1) {
            for (int j = 0; j < 2*CP_NUM; j++)
                out << h_0_L.at<float>(j, 0) << " ";
            for (int j = 0; j < 2*CP_NUM; j++)
                out << h_0_R.at<float>(j, 0) << " ";
            for (int j = 0; j < 2*CP_NUM; j++)
                out << h_0_L.at<float>(j, 0) << " ";
            for (int j = 0; j < 2*CP_NUM; j++)
                out << h_0_R.at<float>(j, 0) << " ";
            out << "\n";
        }
        for (int j = 0; j < 2*CP_NUM; j++)
            out << h_a_L.at<float>(j, 0) << " ";
        for (int j = 0; j < 2*CP_NUM; j++)
            out << h_a_R.at<float>(j, 0) << " ";
        // Kalman filter estimate for next cp iter LOCATION
        h_a_L += dh_diff_L;
        h_a_R += dh_diff_R;
        mc::KalmanStepCP(&h_a_L, &h_a_R, 5.0, 2.0);
        h_a_L.copyTo(h_a_kal_L);
        h_a_R.copyTo(h_a_kal_R);
        for (int j = 0; j < 2*CP_NUM; j++)
            out << h_a_kal_L.at<float>(j, 0) << " ";
        for (int j = 0; j < 2*CP_NUM; j++)
            out << h_a_kal_R.at<float>(j, 0) << " ";

        out << "\n";

//        h_a_L.copyTo(h_a_L_old);
//        h_a_R.copyTo(h_a_R_old);

        ////

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

        cv::Size textSize = getTextSize(text, font_face, font_scale,
                                        font_thickness, &font_baseline);
        font_baseline += font_thickness;

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
            textSize = getTextSize(text_ssim_L, font_face, font_scale_ssim,
                                   font_thickness, &font_baseline);
            cv::Point text_ssim_Lorg(roi_0x_L + (ROI_W - textSize.width)/2, roi_0y_L-3);
            cv::Point text_ssim_Rorg(roi_0x_R + (ROI_W - textSize.width)/2, roi_0y_R-3);
            putText(left_im,  text_ssim_L, text_ssim_Lorg, font_face,
                    font_scale_ssim, cv::Scalar::all(255), font_thickness, 8);
            putText(right_im, text_ssim_R, text_ssim_Rorg, font_face,
                    font_scale_ssim, cv::Scalar::all(255), font_thickness, 8);
        }
        mc::DrawInitBorder(roi_0x_L, roi_0y_L, &frame_L);
        putText(frame_L, text, textOrg, font_face, font_scale,
                cv::Scalar::all(255), font_thickness, 8);
        frame_L.copyTo(mid_im);
        cv::imshow("Original stream, Stabilized stream", current_stable_im);
        if (WRITE)
            cap_write.write(current_stable_im);

        std::cout << "exiting frame # " << frame_num << " after " << iter << " iterations and\t"
                  << static_cast<double>(end-begin)/CLOCKS_PER_SEC << " seconds" << std::endl;

        tot_iters += iter;
        tot_time  += static_cast<double>(end-begin)/CLOCKS_PER_SEC;
        if ((cv::waitKey(1) & 0xFF) == 27 || (cv::waitKey(1) & 0xFF) == 'q'
                || frame_num == end_frame) {
            std::cout << "Program ended by user." << std::endl;
            std::cout << "Average iterations: " << tot_iters/frame_num << " after "
                      << frame_num << " frames" << std::endl;
            std::cout << "Average iters time: " << tot_time/frame_num << std::endl;
            out.close();
            break;
        }
    }

    return 0;
}

