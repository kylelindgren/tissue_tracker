#include "mc_cuda.hpp"

namespace mc_cuda {

//double TPS(double pt1x, double pt1y, double pt2x, double pt2y) {
//    double s_2 = pow(pt1x-pt2x, 2) + pow(pt1y-pt2y, 2);
//    return s_2 == 0 ? 0 : s_2*log(s_2);
//}

//void UpdateDepth(float focal, const cv::Mat h_a_L, const cv::Mat h_a_R, cv::Mat *h_depth) {
//    double disparity;
//    for (int i = 0; i < CP_NUM; i++) {
//        disparity = fabs(h_a_L.at<float>(i, 0) - h_a_R.at<float>(i, 0));
//        h_depth->at<float>(i, 0) = BASELINE * focal / disparity;
//    }
//    return;
//}

//void OnMouse(int event, int x, int y, int, void *center) {
//    if (event == cv::EVENT_LBUTTONDOWN) {
//        cv::Point* p = (cv::Point*) center;
//        p->x = x;
//        p->y = y;
//        std::cout << "Region of interest center: [" << x << ", " << y << "]" << std::endl;
//    }
//    return;
//}

//// function from Richa
//int ComputeJointHistogram(int n_bins, int size_bins, float expected_L[], float expected_R[],
//                          float p_joint_L[], float p_joint_R[],
//                          const cv::cuda::GpuMat frame_L, const cv::cuda::GpuMat frame_R,
//                          const cv::cuda::GpuMat frame_0_L, const cv::cuda::GpuMat frame_0_R) {
//    int u, v, index_L, index_R, flag_error = 0, acc;

//    float p_ref;

//    // zeroing p_joint and acc
//    for (u = 0; u < n_bins*n_bins; u++)
//        p_joint_L[u] = p_joint_R[u] = 0;

//    acc = 0;

//    // computing p_joint between 'current_warp' and 'Template'
//    for (u = 0; u < ROI_W; u++) {
//        for (v = 0; v < ROI_H; v++) {
//            index_L = ((frame_L.at<uchar>(v, u) + 1)/size_bins - 1) +
//                        n_bins*((frame_0_L.at<uchar>(v, u) + 1)/size_bins - 1);
//            index_R = ((frame_R.at<uchar>(v, u) + 1)/size_bins - 1) +
//                        n_bins*((frame_0_R.at<uchar>(v, u) + 1)/size_bins - 1);

//            p_joint_L[index_L]++;
//            p_joint_R[index_R]++;

//            acc++;
//        }
//    }

//    // Normalizing the histogram
//    if (acc > 0) {
//        // left
//        for (u = 0; u < n_bins*n_bins; u++)
//            p_joint_L[u] = p_joint_L[u]/acc;

//        // computing expected intensity values
//        for (u = 0; u < n_bins; u++) {
//            // calcula p_ref
//            p_ref = 0;

//            for (v = 0; v <n_bins; v++)
//                p_ref += p_joint_L[v + n_bins*u];

//            // expected value
//            expected_L[u] = 0;

//            if (p_ref > 0) {
//                for (v = 0; v < n_bins; v++)
//                    expected_L[u] += ((v+1)*p_joint_L[v + n_bins*u]/p_ref);

//                expected_L[u]--;
//            } else {
//                expected_L[u] = static_cast<float>(u);
//            }
//        }
//        // right
//        for (u = 0; u < n_bins*n_bins; u++)
//            p_joint_R[u] = p_joint_R[u]/acc;

//        // computing expected intensity values
//        for (u = 0; u < n_bins; u++) {
//            // calcula p_ref
//            p_ref = 0;

//            for (v = 0; v < n_bins; v++)
//                p_ref += p_joint_R[v + n_bins*u];

//            // expected value
//            expected_R[u] = 0;

//            if (p_ref > 0) {
//                for (v = 0; v < n_bins; v++)
//                    expected_R[u] += ((v+1)*p_joint_R[v + n_bins*u]/p_ref);

//                expected_R[u]--;
//            } else {
//                expected_R[u] = static_cast<float>(u);
//            }
//        }
//    } else {
//        for (u = 0; u < n_bins; u++)
//            expected_L[u] = expected_R[u] = static_cast<float>(u);
//    }

//    return flag_error;
//}

//void ResetExpected(int n_bins, float expected_L[], float expected_R[]) {
//    for (int u = 0; u < n_bins; u++)
//        expected_L[u] = expected_R[u] = static_cast<float>(u);
//}

// function from Richa
void ComputeExpectedImg(const cv::Mat frame_0_L, const cv::Mat frame_0_R,
                        float correction_L[], float correction_R[],
                        float expected_L[], float expected_R[], int n_bins, int size_bins,
                        cv::Ptr<cv::cuda::Filter> filter_x, cv::Ptr<cv::cuda::Filter> filter_y,
                        cv::cuda::GpuMat *frame_comp_L, cv::cuda::GpuMat *frame_comp_R,
                        cv::cuda::GpuMat *gradx_comp_L, cv::cuda::GpuMat *gradx_comp_R,
                        cv::cuda::GpuMat *grady_comp_L, cv::cuda::GpuMat *grady_comp_R) {
    // Calculates intensity value to be added to each intensity value in reference image
    for (int u = 0; u < n_bins; u++) {
        correction_L[u] = size_bins*(expected_L[u] - u);
        correction_R[u] = size_bins*(expected_R[u] - u);
    }

    cv::Mat temp_L, temp_R;
    frame_comp_L->download(temp_L);
    frame_comp_R->download(temp_R);
    // Correcting template
    for (int v = 0; v < ROI_H; v++)
        for (int u = 0; u < ROI_W; u++) {
            temp_L.at<uchar>(v, u) = frame_0_L.at<uchar>(v, u) +
                    cvRound(correction_L[cvRound(static_cast<float>
                                                 (frame_0_L.at<uchar>(v, u)/size_bins))]);
            temp_R.at<uchar>(v, u) = frame_0_R.at<uchar>(v, u) +
                    cvRound(correction_R[cvRound(static_cast<float>
                                                 (frame_0_R.at<uchar>(v, u)/size_bins))]);
        }
    frame_comp_L->upload(temp_L);
    frame_comp_R->upload(temp_R);

    // Re-computing Gradient
    filter_x->apply(*frame_comp_L, *gradx_comp_L);
    filter_y->apply(*frame_comp_L, *grady_comp_L);
    filter_x->apply(*frame_comp_R, *gradx_comp_R);
    filter_y->apply(*frame_comp_R, *grady_comp_R);
}

void Jacobian(const cv::Mat I, cv::Ptr<cv::cuda::Filter> sobel_x,
              cv::Ptr<cv::cuda::Filter> sobel_y,  const cv::Mat MK, cv::Mat *J) {
    cv::cuda::GpuMat gradx_c = cv::cuda::createContinuous(ROI_H, ROI_W, CV_32FC1);
    cv::cuda::GpuMat grady_c = cv::cuda::createContinuous(ROI_H, ROI_W, CV_32FC1);
//    cv::cuda::GpuMat gradx_c_(ROI_H, ROI_W, CV_32FC1);
//    cv::cuda::GpuMat grady_c_(ROI_H, ROI_W, CV_32FC1);
    cv::cuda::GpuMat I_c;
//    cv::cuda::GpuMat MK_c(PIXELS, CP_NUM, CV_32FC1);
//    cv::cuda::GpuMat J_L_c(PIXELS, CP_NUM, CV_32FC1);
//    cv::cuda::GpuMat J_R_c(PIXELS, CP_NUM, CV_32FC1);
    I_c.upload(I);
//    MK_c.upload(MK);
    sobel_x->apply(I_c, gradx_c);
    sobel_y->apply(I_c, grady_c);
    cv::Mat gradx, grady;
    gradx_c.download(gradx);
    grady_c.download(grady);
//    cv::Mat J_L = J->colRange(0, CP_NUM).rowRange(0, PIXELS);
//    cv::Mat J_R = J->colRange(CP_NUM, CP_NUM*2).rowRange(0, PIXELS);
    float tempx, tempy;
    for (int r = 0; r < PIXELS; r++) {
        tempx = gradx.at<float>(static_cast<int>(r/ROI_W), r%ROI_W);
        tempy = grady.at<float>(static_cast<int>(r/ROI_W), r%ROI_W);
        for (int c = 0; c < CP_NUM*2; c++) {
            if (c < CP_NUM)
                J->at<float>(r, c) = MK.at<float>(r, c%CP_NUM)*tempx;
            else
                J->at<float>(r, c) = MK.at<float>(r, c%CP_NUM)*tempy;
        }
    }

//    cv::cuda::GpuMat gradx_1d_c = gradx_c.reshape(1, PIXELS);
//    cv::cuda::GpuMat grady_1d_c = grady_c.reshape(1, PIXELS);
//    gradx_1d_c.convertTo(gradx_1d_c, CV_32FC1);
//    cv::cuda::GpuMat gradx_1d_c_ = cv::cuda::GpuMat(PIXELS, 1, CV_32FC1, &gradx_1d_c);

//    cv::cuda::GpuMat mat = cv::cuda::GpuMat(PIXELS, 4, CV_32FC1);
//    cv::cuda::GpuMat temp1 = cv::cuda::GpuMat(cv::cuda::GpuMat *mat, cv::Rect(0, PIXELS, 0, 1));

//    cv::cuda::gemm(MK_c, gradx_1d_c_, 1, J_L_c, 0, J_L_c);  // doesn't do element wise...
//    cv::cuda::multiply(MK_c, gradx_1d_c_, J_R_c);

//    J_L_c.download(J_L);
//    J_R_c.download(J_R);
    return;
}

//void drawGrid(const cv::cuda::GpuMat MK_L, const cv::cuda::GpuMat h, cv::cuda::GpuMat *image) {
//    int step_x = floor(ROI_W/(CP_NUM+1));
//    int step_y = floor(ROI_H/(CP_NUM+1));
//    int k;
//    float x1, x2, y1, y2;
//    cv::cuda::GpuMat mat_x1(1, 1, CV_32F, &x1), mat_x2(1, 1, CV_32F, &x2);
//    cv::cuda::GpuMat mat_y1(1, 1, CV_32F, &y1), mat_y2(1, 1, CV_32F, &y2);
//    cv::cuda::GpuMat hx = cv::cuda::GpuMat::zeros(CP_NUM, 1, CV_32FC1);
//    cv::cuda::GpuMat hy = cv::cuda::GpuMat::zeros(CP_NUM, 1, CV_32FC1);
//    hx = h(cv::Range(0, CP_NUM), cv::Range::all());
//    hy = h(cv::Range(CP_NUM, CP_NUM*2), cv::Range::all());
//    int line_thick = 1;
//    int line_type = CV_AA;  // CV_AA, 4, 8

//    // horizontal lines
//    for (int r = 0; r <= CP_NUM; r++)
//        for (int c = 0; c < CP_NUM; c++) {
//            k = r*ROI_W*step_y + c*step_x;
//            mat_x1 = MK_L.row(k)*hx;
//            mat_y1 = MK_L.row(k)*hy;
//            if (r == CP_NUM) {
//                mat_x2 = MK_L.row(k+step_x-1)*hx;
//                mat_y2 = MK_L.row(k+step_x-1)*hy;
//            } else {
//                mat_x2 = MK_L.row(k+step_x)*hx;
//                mat_y2 = MK_L.row(k+step_x)*hy;
//            }
//            line(*image, cv::Point(round(x1), round(y1)), cv::Point(round(x2), round(y2)),
//                 cv::Scalar(255, 255, 255), line_thick, line_type);
//        }

//    // vertical lines
//    for (int c = 0; c <= CP_NUM; c++)
//        for (int r = 0; r < CP_NUM; r++) {
//            k = r*ROI_W*step_y + c*step_x;
//            mat_x1 = MK_L.row(k)*hx;
//            mat_y1 = MK_L.row(k)*hy;
//            if (c == CP_NUM) {
//                mat_x2 = MK_L.row(k+(step_y-1)*ROI_W)*hx;
//                mat_y2 = MK_L.row(k+(step_y-1)*ROI_W)*hy;
//            } else {
//                mat_x2 = MK_L.row(k+step_y*ROI_W)*hx;
//                mat_y2 = MK_L.row(k+step_y*ROI_W)*hy;
//            }
//            line(*image, cv::Point(round(x1), round(y1)), cv::Point(round(x2), round(y2)),
//                 cv::Scalar(255, 255, 255), line_thick, line_type);
//        }
//}

void AffineTrans(const cv::Mat MK_L, const cv::Mat MK_R, const cv::Mat X_L, const cv::Mat X_R,
                 const cv::Mat h_a, bool left, cv::Mat *image) {
    cv::Mat warp_mat_inv;
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];
    float x1, x2, x3, x4, y1, y2, y3, y4;
    cv::Mat mat_x1(1, 1, CV_32F, &x1), mat_x2(1, 1, CV_32F, &x2);
    cv::Mat mat_x3(1, 1, CV_32F, &x3), mat_x4(1, 1, CV_32F, &x4);
    cv::Mat mat_y1(1, 1, CV_32F, &y1), mat_y2(1, 1, CV_32F, &y2);
    cv::Mat mat_y3(1, 1, CV_32F, &y3), mat_y4(1, 1, CV_32F, &y4);
    cv::Mat hx = cv::Mat::zeros(CP_NUM, 1, CV_32FC1);
    cv::Mat hy = cv::Mat::zeros(CP_NUM, 1, CV_32FC1);
    hx = h_a(cv::Range(0, CP_NUM), cv::Range::all());
    hy = h_a(cv::Range(CP_NUM, CP_NUM*2), cv::Range::all());

    if (left) {
        mat_x1 = MK_L.row(0)*hx;
        mat_x2 = MK_L.row(ROI_W-1)*hx;
        mat_x3 = MK_L.row(ROI_W*(ROI_H-1))*hx;
    //    mat_x4 = MK_L.row(ROI_W*ROI_H-1)*hx;
        mat_y1 = MK_L.row(0)*hy;
        mat_y2 = MK_L.row(ROI_W-1)*hy;
        mat_y3 = MK_L.row(ROI_W*(ROI_H-1))*hy;
    //    mat_y4 = MK_L.row(ROI_W*ROI_H-1)*hy;

        srcTri[0] = cv::Point2f(static_cast<float>(X_L.at<ushort>(0, 0)),
                                static_cast<float>(X_L.at<ushort>(0, 1)));
        srcTri[1] = cv::Point2f(static_cast<float>(X_L.at<ushort>(ROI_W-1, 0)),
                                static_cast<float>(X_L.at<ushort>(ROI_W-1, 1)));
        srcTri[2] = cv::Point2f(static_cast<float>(X_L.at<ushort>(ROI_W*(ROI_H-1), 0)),
                                static_cast<float>(X_L.at<ushort>(ROI_W*(ROI_H-1), 1)));
//    srcTri[3] = cv::Point2f(static_cast<float>(X_L.at<ushort>(ROI_W*ROI_H-1, 0)),
//                                static_cast<float>(X_L.at<ushort>(ROI_W*ROI_H-1, 1)));
    } else {
        mat_x1 = MK_R.row(0)*hx;
        mat_x2 = MK_R.row(ROI_W-1)*hx;
        mat_x3 = MK_R.row(ROI_W*(ROI_H-1))*hx;
    //    mat_x4 = MK_R.row(ROI_W*ROI_H-1)*hx;
        mat_y1 = MK_R.row(0)*hy;
        mat_y2 = MK_R.row(ROI_W-1)*hy;
        mat_y3 = MK_R.row(ROI_W*(ROI_H-1))*hy;
    //    mat_y4 = MK_R.row(ROI_W*ROI_H-1)*hy;

        srcTri[0] = cv::Point2f(static_cast<float>(X_R.at<ushort>(0, 0)),
                                static_cast<float>(X_R.at<ushort>(0, 1)));
        srcTri[1] = cv::Point2f(static_cast<float>(X_R.at<ushort>(ROI_W-1, 0)),
                                static_cast<float>(X_R.at<ushort>(ROI_W-1, 1)));
        srcTri[2] = cv::Point2f(static_cast<float>(X_R.at<ushort>(ROI_W*(ROI_H-1), 0)),
                                static_cast<float>(X_R.at<ushort>(ROI_W*(ROI_H-1), 1)));
    //    srcTri[3] = cv::Point2f(static_cast<float>(X_R.at<ushort>(ROI_W*ROI_H-1, 0),
//                                static_cast<float>(X_R.at<ushort>(ROI_W*ROI_H-1, 1)));
    }

    dstTri[0] = cv::Point2f(static_cast<float>(x1), static_cast<float>(y1));
    dstTri[1] = cv::Point2f(static_cast<float>(x2), static_cast<float>(y2));
    dstTri[2] = cv::Point2f(static_cast<float>(x3), static_cast<float>(y3));
//    dstTri[3] = cv::Point2f(static_cast<float>(x4), static_cast<float>(y4));
//    std::cout << dstTri[0] << " " << dstTri[1] << " " << dstTri[2] << std::endl;

    warp_mat_inv = cv::getAffineTransform(dstTri, srcTri);
//    warp_mat_inv = cv::getPerspectiveTransform(dstTri, srcTri);
//    cv::warpAffine(*image, *image, warp_mat_inv, image->size());
    cv::cuda::warpAffine(*image, *image, warp_mat_inv, image->size());
//    cv::warpPerspective(image, image, warp_mat_inv, image.size());
}

//double CalcDepth(double disp) {
////    cv::pow(test, 2, test);
//    return -8.4171e-05*pow(disp,3) + 0.037*pow(disp,2) + -6.0613*disp + 478.7784;
//}


}  // namespace mc_cuda
