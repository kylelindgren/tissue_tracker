#include "mc.hpp"

namespace mc {

double TPS(double pt1x, double pt1y, double pt2x, double pt2y) {
    double s_2 = pow(pt1x-pt2x, 2) + pow(pt1y-pt2y, 2);
    return s_2 == 0 ? 0 : s_2*log(s_2);
}

void CalcMXT(const cv::Mat h_0_L, const cv::Mat h_0_R, cv::Mat const frame_0_L,
             const cv::Mat frame_0_R, const cv::Mat X_L, cv::Mat *X_R,
             cv::Mat *M_L, cv::Mat *M_R, cv::Mat *T_L, cv::Mat *T_R) {
    double x_pos, y_pos;
    int offx;
    int offy;
    for (int r = 0; r < PIXELS; r++) {
        x_pos = X_L.at<ushort>(r, 0);
        y_pos = X_L.at<ushort>(r, 1);
        T_L->at<float>(r, 0) = frame_0_L.at<uchar>(y_pos, x_pos);
        for (int c = 0; c <= CP_NUM; c++) {  // fill M
            if (c != CP_NUM) {
                M_L->at<float>(r, c) = TPS(h_0_L.at<float>(c, 0),
                                          h_0_L.at<float>(c+CP_NUM, 0), x_pos, y_pos);
            } else {
                M_L->at<float>(r, c+0) = 1;
                M_L->at<float>(r, c+1) = x_pos;
                M_L->at<float>(r, c+2) = y_pos;
            }
        }
    }
    offx = X_R->at<ushort>(0, 0);
    offy = X_R->at<ushort>(0, 1);
    for (int r = 0; r < PIXELS; r++) {
        x_pos = static_cast<int>(r%ROI_W)+offx;
        y_pos = static_cast<int>(r/ROI_W)+offy;
        X_R->at<ushort>(r, 0) = x_pos;  // fill X and T
        X_R->at<ushort>(r, 1) = y_pos;
        T_R->at<float>(r, 0) = frame_0_R.at<uchar>(y_pos, x_pos);
        for (int c = 0; c <= CP_NUM; c++) {
            if (c != CP_NUM) {
                M_R->at<float>(r, c) = TPS(h_0_R.at<float>(c, 0),
                                          h_0_R.at<float>(c+CP_NUM, 0), x_pos, y_pos);
            } else {
                M_R->at<float>(r, c+0) = 1;
                M_R->at<float>(r, c+1) = x_pos;
                M_R->at<float>(r, c+2) = y_pos;
            }
        }
    }
    return;
}

void CalcK(const cv::Mat h, cv::Mat *K) {
    for (int r = 0 ; r < CP_NUM; r++)
        for (int c = 0; c <= CP_NUM; c++)
            if (c > r || c == CP_NUM) {  // K symmetric so only calculate upper right, copy values
                if (c == CP_NUM) {
                    K->at<float>(r, c+0) = K->at<float>(c+0, r) = 1;
                    K->at<float>(r, c+1) = K->at<float>(c+1, r) = h.at<float>(r, 0);
                    K->at<float>(r, c+2) = K->at<float>(c+2, r) = h.at<float>(r+CP_NUM, 0);
                } else {
                    K->at<float>(r, c) = K->at<float>(c, r) = TPS(h.at<float>(r, 0),
                        h.at<float>(r+CP_NUM, 0), h.at<float>(c, 0), h.at<float>(c+CP_NUM, 0));
                }
            }
    return;
}

void UpdateDepth(float focal, const cv::Mat h_a_L, const cv::Mat h_a_R, cv::Mat *h_depth) {
    double disparity;
    for (int i = 0; i < CP_NUM; i++) {
        disparity = fabs(h_a_L.at<float>(i, 0) - h_a_R.at<float>(i, 0));
        h_depth->at<float>(i, 0) = BASELINE * focal / disparity;
    }
    return;
}

void OnMouse(int event, int x, int y, int, void *center) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Point* p = (cv::Point*) center;
        p->x = x;
        p->y = y;
        std::cout << "Region of interest center: [" << x << ", " << y << "]" << std::endl;
    }
    return;
}

void DrawROIBorder(const cv::Mat MK_L, const cv::Mat MK_R, const cv::Mat h,
                   bool left, cv::Mat *frame) {
    int line_thick = 1, line_type = CV_AA;
    cv::Scalar color = cv::Scalar(255, 255, 255);

    float ulx, uly, urx, ury, llx, lly, lrx, lry;
    cv::Mat mat_ulx(1, 1, CV_32F, &ulx), mat_uly(1, 1, CV_32F, &uly);
    cv::Mat mat_urx(1, 1, CV_32F, &urx), mat_ury(1, 1, CV_32F, &ury);
    cv::Mat mat_llx(1, 1, CV_32F, &llx), mat_lly(1, 1, CV_32F, &lly);
    cv::Mat mat_lrx(1, 1, CV_32F, &lrx), mat_lry(1, 1, CV_32F, &lry);

    cv::Mat hx = cv::Mat::zeros(CP_NUM, 1, CV_32FC1);
    cv::Mat hy = cv::Mat::zeros(CP_NUM, 1, CV_32FC1);
    hx = h(cv::Range(0, CP_NUM), cv::Range::all());
    hy = h(cv::Range(CP_NUM, CP_NUM*2), cv::Range::all());

    if (left) {
        mat_ulx = MK_L.row(0)*hx;
        mat_uly = MK_L.row(0)*hy;
        mat_urx = MK_L.row(ROI_W-1)*hx;
        mat_ury = MK_L.row(ROI_W-1)*hy;
        mat_llx = MK_L.row(PIXELS-ROI_W+1)*hx;
        mat_lly = MK_L.row(PIXELS-ROI_W+1)*hy;
        mat_lrx = MK_L.row(PIXELS-1)*hx;
        mat_lry = MK_L.row(PIXELS-1)*hy;
    } else {
        mat_ulx = MK_R.row(0)*hx;
        mat_uly = MK_R.row(0)*hy;
        mat_urx = MK_R.row(ROI_W-1)*hx;
        mat_ury = MK_R.row(ROI_W-1)*hy;
        mat_llx = MK_R.row(PIXELS-ROI_W+1)*hx;
        mat_lly = MK_R.row(PIXELS-ROI_W+1)*hy;
        mat_lrx = MK_R.row(PIXELS-1)*hx;
        mat_lry = MK_R.row(PIXELS-1)*hy;
    }
    line(*frame, cv::Point(round(ulx), round(uly)),
                cv::Point(round(urx), round(ury)), color, line_thick, line_type);
    line(*frame, cv::Point(round(llx), round(lly)),
                cv::Point(round(lrx), round(lry)), color, line_thick, line_type);
    line(*frame, cv::Point(round(ulx), round(uly)),
                cv::Point(round(llx), round(lly)), color, line_thick, line_type);
    line(*frame, cv::Point(round(urx), round(ury)),
                cv::Point(round(lrx), round(lry)), color, line_thick, line_type);
    return;
}

void DrawInitBorder(int roi_0x_L, int roi_0y_L, cv::Mat *frame) {
    int line_thick = 1, line_type = CV_AA;
    cv::Scalar color = cv::Scalar(255, 255, 255);
    int lx = roi_0x_L, rx = lx + ROI_W;
    int ty = roi_0y_L, by = ty + ROI_H;

    line(*frame, cv::Point(lx, ty), cv::Point(rx, ty), color, line_thick, line_type);
    line(*frame, cv::Point(lx, by), cv::Point(rx, by), color, line_thick, line_type);
    line(*frame, cv::Point(lx, ty), cv::Point(lx, by), color, line_thick, line_type);
    line(*frame, cv::Point(rx, ty), cv::Point(rx, by), color, line_thick, line_type);

    return;
}

// function from Richa
int ComputeJointHistogram(int n_bins, int size_bins, float expected_L[], float expected_R[],
                          float p_joint_L[], float p_joint_R[],
                          const cv::Mat frame_L, const cv::Mat frame_R,
                          const cv::Mat frame_0_L, const cv::Mat frame_0_R) {
    int u, v, index_L, index_R, flag_error = 0, acc;

    float p_ref;

    // zeroing p_joint and acc
    for (u = 0; u < n_bins*n_bins; u++)
        p_joint_L[u] = p_joint_R[u] = 0;

    acc = 0;

    // computing p_joint between 'current_warp' and 'Template'
    for (u = 0; u < ROI_W; u++) {
        for (v = 0; v < ROI_H; v++) {
            index_L = ((frame_L.at<uchar>(v, u) + 1)/size_bins - 1) +
                        n_bins*((frame_0_L.at<uchar>(v, u) + 1)/size_bins - 1);
            index_R = ((frame_R.at<uchar>(v, u) + 1)/size_bins - 1) +
                        n_bins*((frame_0_R.at<uchar>(v, u) + 1)/size_bins - 1);

            p_joint_L[index_L]++;
            p_joint_R[index_R]++;

            acc++;
        }
    }

    // Normalizing the histogram
    if (acc > 0) {
        // left
        for (u = 0; u < n_bins*n_bins; u++)
            p_joint_L[u] = p_joint_L[u]/acc;

        // computing expected intensity values
        for (u = 0; u < n_bins; u++) {
            // calcula p_ref
            p_ref = 0;

            for (v = 0; v <n_bins; v++)
                p_ref += p_joint_L[v + n_bins*u];

            // expected value
            expected_L[u] = 0;

            if (p_ref > 0) {
                for (v = 0; v < n_bins; v++)
                    expected_L[u] += ((v+1)*p_joint_L[v + n_bins*u]/p_ref);

                expected_L[u]--;
            } else {
                expected_L[u] = static_cast<float>(u);
            }
        }
        // right
        for (u = 0; u < n_bins*n_bins; u++)
            p_joint_R[u] = p_joint_R[u]/acc;

        // computing expected intensity values
        for (u = 0; u < n_bins; u++) {
            // calcula p_ref
            p_ref = 0;

            for (v = 0; v < n_bins; v++)
                p_ref += p_joint_R[v + n_bins*u];

            // expected value
            expected_R[u] = 0;

            if (p_ref > 0) {
                for (v = 0; v < n_bins; v++)
                    expected_R[u] += ((v+1)*p_joint_R[v + n_bins*u]/p_ref);

                expected_R[u]--;
            } else {
                expected_R[u] = static_cast<float>(u);
            }
        }
    } else {
        for (u = 0; u < n_bins; u++)
            expected_L[u] = expected_R[u] = static_cast<float>(u);
    }

    return flag_error;
}

void ResetExpected(int n_bins, float expected_L[], float expected_R[]) {
    for (int u = 0; u < n_bins; u++)
        expected_L[u] = expected_R[u] = static_cast<float>(u);
}

// function from Richa
void ComputeExpectedImg(const cv::Mat frame_0_L, const cv::Mat frame_0_R,
                        float correction_L[], float correction_R[],
                        float expected_L[], float expected_R[], int n_bins, int size_bins,
                        cv::Mat *frame_comp_L, cv::Mat *frame_comp_R,
                        cv::Mat *gradx_comp_L, cv::Mat *gradx_comp_R,
                        cv::Mat *grady_comp_L, cv::Mat *grady_comp_R) {
    // Calculates intensity value to be added to each intensity value in reference image
    for (int u = 0; u < n_bins; u++) {
        correction_L[u] = size_bins*(expected_L[u] - u);
        correction_R[u] = size_bins*(expected_R[u] - u);
    }

    // Correcting template
    for (int v = 0; v < ROI_H; v++)
        for (int u = 0; u < ROI_W; u++) {
            frame_comp_L->at<uchar>(v, u) = frame_0_L.at<uchar>(v, u) +
                    cvRound(correction_L[cvRound(static_cast<float>
                                                 (frame_0_L.at<uchar>(v, u)/size_bins))]);
            frame_comp_R->at<uchar>(v, u) = frame_0_R.at<uchar>(v, u) +
                    cvRound(correction_R[cvRound(static_cast<float>
                                                 (frame_0_R.at<uchar>(v, u)/size_bins))]);
        }

    // Re-computing Gradient
    Sobel(*frame_comp_L, *gradx_comp_L, CV_32F, 1, 0, KSIZE);
    Sobel(*frame_comp_L, *grady_comp_L, CV_32F, 0, 1, KSIZE);
    Sobel(*frame_comp_R, *gradx_comp_R, CV_32F, 1, 0, KSIZE);
    Sobel(*frame_comp_R, *grady_comp_R, CV_32F, 0, 1, KSIZE);
}

void UpdateJ_0(const cv::Mat MK_L, const cv::Mat MK_R,
               const cv::Mat X_L, const cv::Mat X_R,
               const cv::Mat gradx_comp_L, const cv::Mat gradx_comp_R,
               const cv::Mat grady_comp_L, const cv::Mat grady_comp_R,
               cv::Mat *J_0_L, cv::Mat *J_0_R) {
///*
    float tempx_L, tempy_L, tempx_R, tempy_R;
    for (int r = 0; r < PIXELS; r++) {
        tempx_L = gradx_comp_L.at<float>
                (X_L.at<ushort>(r, 1), X_L.at<ushort>(r, 0));
        tempy_L = grady_comp_L.at<float>
                (X_L.at<ushort>(r, 1), X_L.at<ushort>(r, 0));
        tempx_R = gradx_comp_R.at<float>
                (X_R.at<ushort>(r, 1), X_R.at<ushort>(r, 0));
        tempy_R = grady_comp_R.at<float>
                (X_R.at<ushort>(r, 1), X_R.at<ushort>(r, 0));
        for (int c = 0; c < CP_NUM*2; c++) {
            if (c < CP_NUM) {
                J_0_L->at<float>(r, c) = MK_L.at<float>(r, c%CP_NUM)*tempx_L;
                J_0_R->at<float>(r, c) = MK_R.at<float>(r, c%CP_NUM)*tempx_R;
            } else {
                J_0_L->at<float>(r, c) = MK_L.at<float>(r, c%CP_NUM)*tempy_L;
                J_0_R->at<float>(r, c) = MK_R.at<float>(r, c%CP_NUM)*tempy_R;
            }
        }
    }
//*/

/*
    cv::Mat temp_L = cv::Mat(1, 1, CV_32FC1);
    cv::Mat temp_R = cv::Mat(1, 1, CV_32FC1);
    cv::Mat mat_temp = cv::Mat::zeros(CP_NUM, 1, CV_32FC1);
    for (int r = 0; r < PIXELS; r++)
        for (int c = 0; c < CP_NUM*2; c++) {
            mat_temp = cv::Scalar(0);
            mat_temp.at<float>(c%CP_NUM, 0) = 1;
            temp_L = MK_L.row(r)*mat_temp;
            temp_R = MK_R.row(r)*mat_temp;
            if (c < CP_NUM) {
                J_0_L->at<float>(r, c) = temp_L.at<float>(0, 0)*gradx_comp_L.at<float>
                        (X_L.at<ushort>(r, 1), X_L.at<ushort>(r, 0));
                J_0_R->at<float>(r, c) = temp_R.at<float>(0, 0)*gradx_comp_R.at<float>
                        (X_R.at<ushort>(r, 1), X_R.at<ushort>(r, 0));
            } else {
                J_0_L->at<float>(r, c) = temp_L.at<float>(0, 0)*grady_comp_L.at<float>
                        (X_L.at<ushort>(r, 1), X_L.at<ushort>(r, 0));
                J_0_R->at<float>(r, c) = temp_R.at<float>(0, 0)*grady_comp_R.at<float>
                        (X_R.at<ushort>(r, 1), X_R.at<ushort>(r, 0));
            }
        }
//*/
}

void Jacobian(const cv::Mat I, const cv::Mat MK, cv::Mat *J) {
/*
//    clock_t begin = clock();
    cv::Mat gradx, grady;
    cv::Sobel(I, gradx, CV_32F, 1, 0, KSIZE);
    cv::Sobel(I, grady, CV_32F, 0, 1, KSIZE);
//    cv::Mat J_L = cv::Mat::zeros(PIXELS, CP_NUM, CV_32FC1);
//    cv::Mat J_R = cv::Mat::zeros(PIXELS, CP_NUM, CV_32FC1);
    cv::Mat J_L = J->colRange(0, CP_NUM).rowRange(0, PIXELS);
    cv::Mat J_R = J->colRange(CP_NUM, CP_NUM*2).rowRange(0, PIXELS);
//    cv::Mat gradx_1d = gradx.reshape(1, PIXELS);
//    cv::multiply(MK, gradx_1d, J_L);
    float tempx, tempy;
//    std::cout << "init\t" << static_cast<double>(clock()-begin)/CLOCKS_PER_SEC
//              << " seconds" << std::endl;
//    begin = clock();
    for (int r = 0; r < PIXELS; r++) {
//        tempx = gradx.at<float>(static_cast<int>(r/ROI_W), r%ROI_W);
//        tempy = grady.at<float>(static_cast<int>(r/ROI_W), r%ROI_W);
        J_L.row(r) = MK.row(r) * gradx.at<float>(static_cast<int>(r/ROI_W), r%ROI_W);
        J_R.row(r) = MK.row(r) * grady.at<float>(static_cast<int>(r/ROI_W), r%ROI_W);
//        J_L.row(r) = MK.row(r)*tempx;
//        J_R.row(r) = MK.row(r)*tempy;
    }
//    std::cout << "loop\t" << static_cast<double>(clock()-begin)/CLOCKS_PER_SEC
//              << " seconds" << std::endl << std::endl;
//*/

///*
//    clock_t begin = clock();
    cv::Mat gradx, grady;
    cv::Sobel(I, gradx, CV_32F, 1, 0, KSIZE);
    cv::Sobel(I, grady, CV_32F, 0, 1, KSIZE);
    float tempx, tempy;
//    std::cout << "init\t" << static_cast<double>(clock()-begin)/CLOCKS_PER_SEC
//              << " seconds" << std::endl;
//    begin = clock();
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
//    std::cout << "loop\t" << static_cast<double>(clock()-begin)/CLOCKS_PER_SEC
//              << " seconds" << std::endl << std::endl;
//*/


/*
    cv::Mat gradx, grady;
    cv::Mat temp = cv::Mat(1, 1, CV_32FC1);
    cv::Mat mat_temp = cv::Mat(CP_NUM, 1, CV_32FC1);
    cv::Sobel(I, gradx, CV_32F, 1, 0, KSIZE);
    cv::Sobel(I, grady, CV_32F, 0, 1, KSIZE);
    cv::Mat grad_j = cv::Mat(1, 1, CV_32FC1);
    for (int r = 0; r < PIXELS; r++) {
        for (int c = 0; c < CP_NUM*2; c++) {
            mat_temp = cv::Scalar(0);
            mat_temp.at<float>(c%CP_NUM, 0) = 1;
            temp = MK.row(r)*mat_temp;
            if (c < CP_NUM)
                grad_j = temp.at<float>(0, 0)*gradx.at<float>(static_cast<int>(r/ROI_W), r%ROI_W);
            else
                grad_j = temp.at<float>(0, 0)*grady.at<float>(static_cast<int>(r/ROI_W), r%ROI_W);

            J->at<float>(r, c) = grad_j.at<float>(0, 0);
        }
    }
//*/
    return;

}

void drawGrid(const cv::Mat MK_L, const cv::Mat h, cv::Mat *image) {
    int step_x = floor(ROI_W/(CP_NUM+1));
    int step_y = floor(ROI_H/(CP_NUM+1));
    int k;
    float x1, x2, y1, y2;
    cv::Mat mat_x1(1, 1, CV_32F, &x1), mat_x2(1, 1, CV_32F, &x2);
    cv::Mat mat_y1(1, 1, CV_32F, &y1), mat_y2(1, 1, CV_32F, &y2);
    cv::Mat hx = cv::Mat::zeros(CP_NUM, 1, CV_32FC1);
    cv::Mat hy = cv::Mat::zeros(CP_NUM, 1, CV_32FC1);
    hx = h(cv::Range(0, CP_NUM), cv::Range::all());
    hy = h(cv::Range(CP_NUM, CP_NUM*2), cv::Range::all());
    int line_thick = 1;
    int line_type = CV_AA;  // CV_AA, 4, 8

    // horizontal lines
    for (int r = 0; r <= CP_NUM; r++)
        for (int c = 0; c < CP_NUM; c++) {
            k = r*ROI_W*step_y + c*step_x;
            mat_x1 = MK_L.row(k)*hx;
            mat_y1 = MK_L.row(k)*hy;
            if (r == CP_NUM) {
                mat_x2 = MK_L.row(k+step_x-1)*hx;
                mat_y2 = MK_L.row(k+step_x-1)*hy;
            } else {
                mat_x2 = MK_L.row(k+step_x)*hx;
                mat_y2 = MK_L.row(k+step_x)*hy;
            }
            line(*image, cv::Point(round(x1), round(y1)), cv::Point(round(x2), round(y2)),
                 cv::Scalar(255, 255, 255), line_thick, line_type);
        }

    // vertical lines
    for (int c = 0; c <= CP_NUM; c++)
        for (int r = 0; r < CP_NUM; r++) {
            k = r*ROI_W*step_y + c*step_x;
            mat_x1 = MK_L.row(k)*hx;
            mat_y1 = MK_L.row(k)*hy;
            if (c == CP_NUM) {
                mat_x2 = MK_L.row(k+(step_y-1)*ROI_W)*hx;
                mat_y2 = MK_L.row(k+(step_y-1)*ROI_W)*hy;
            } else {
                mat_x2 = MK_L.row(k+step_y*ROI_W)*hx;
                mat_y2 = MK_L.row(k+step_y*ROI_W)*hy;
            }
            line(*image, cv::Point(round(x1), round(y1)), cv::Point(round(x2), round(y2)),
                 cv::Scalar(255, 255, 255), line_thick, line_type);
        }
}

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
        mat_x4 = MK_L.row(ROI_W*ROI_H-1)*hx;
        mat_y1 = MK_L.row(0)*hy;
        mat_y2 = MK_L.row(ROI_W-1)*hy;
        mat_y3 = MK_L.row(ROI_W*(ROI_H-1))*hy;
        mat_y4 = MK_L.row(ROI_W*ROI_H-1)*hy;

        srcTri[0] = cv::Point2f(static_cast<float>(X_L.at<ushort>(0, 0)),
                                static_cast<float>(X_L.at<ushort>(0, 1)));
        srcTri[1] = cv::Point2f(static_cast<float>(X_L.at<ushort>(ROI_W-1, 0)),
                                static_cast<float>(X_L.at<ushort>(ROI_W-1, 1)));
        srcTri[2] = cv::Point2f(static_cast<float>(X_L.at<ushort>(ROI_W*(ROI_H-1), 0)),
                                static_cast<float>(X_L.at<ushort>(ROI_W*(ROI_H-1), 1)));
        srcTri[3] = cv::Point2f(static_cast<float>(X_L.at<ushort>(ROI_W*ROI_H-1, 0)),
                                static_cast<float>(X_L.at<ushort>(ROI_W*ROI_H-1, 1)));
    } else {
        mat_x1 = MK_R.row(0)*hx;
        mat_x2 = MK_R.row(ROI_W-1)*hx;
        mat_x3 = MK_R.row(ROI_W*(ROI_H-1))*hx;
        mat_x4 = MK_R.row(ROI_W*ROI_H-1)*hx;
        mat_y1 = MK_R.row(0)*hy;
        mat_y2 = MK_R.row(ROI_W-1)*hy;
        mat_y3 = MK_R.row(ROI_W*(ROI_H-1))*hy;
        mat_y4 = MK_R.row(ROI_W*ROI_H-1)*hy;

        srcTri[0] = cv::Point2f(static_cast<float>(X_R.at<ushort>(0, 0)),
                                static_cast<float>(X_R.at<ushort>(0, 1)));
        srcTri[1] = cv::Point2f(static_cast<float>(X_R.at<ushort>(ROI_W-1, 0)),
                                static_cast<float>(X_R.at<ushort>(ROI_W-1, 1)));
        srcTri[2] = cv::Point2f(static_cast<float>(X_R.at<ushort>(ROI_W*(ROI_H-1), 0)),
                                static_cast<float>(X_R.at<ushort>(ROI_W*(ROI_H-1), 1)));
        srcTri[3] = cv::Point2f(static_cast<float>(X_R.at<ushort>(ROI_W*ROI_H-1, 0)),
                                static_cast<float>(X_R.at<ushort>(ROI_W*ROI_H-1, 1)));
    }

    dstTri[0] = cv::Point2f(static_cast<float>(x1), static_cast<float>(y1));
    dstTri[1] = cv::Point2f(static_cast<float>(x2), static_cast<float>(y2));
    dstTri[2] = cv::Point2f(static_cast<float>(x3), static_cast<float>(y3));
    dstTri[3] = cv::Point2f(static_cast<float>(x4), static_cast<float>(y4));
//    std::cout << dstTri[0] << " " << dstTri[1] << " " << dstTri[2] << std::endl;

//    warp_mat_inv = cv::getAffineTransform(dstTri, srcTri);
//    cv::warpAffine(*image, *image, warp_mat_inv, image->size());
    warp_mat_inv = cv::getPerspectiveTransform(dstTri, srcTri);
    cv::warpPerspective(*image, *image, warp_mat_inv, image->size());
}


int MatchFeatures(const cv::Mat left, const cv::Mat right,
            std::vector<cv::Point2f> *left_features, std::vector<cv::Point2f> *right_features) {
    cv::Point2f point1;
    cv::Point2f point2;
    int found_features;
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    detector->setHessianThreshold(minHessian);
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    detector->detectAndCompute(left,  cv::Mat(), keypoints_1, descriptors_1);
    detector->detectAndCompute(right, cv::Mat(), keypoints_2, descriptors_2);

    //-- Step 2: Matching descriptor vectors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
//    printf("-- Max dist : %f \n", max_dist );
//    printf("-- Min dist : %f \n", min_dist );
    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2*min_dist, 0.02))
            good_matches.push_back(matches[i]);
    }
    //-- Draw only "good" matches
    cv::Mat img_matches;
    drawMatches(left, keypoints_1, right, keypoints_2, good_matches, img_matches,
                cv::Scalar::all(-1), cv::Scalar::all(-1),
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    found_features = static_cast<int>(good_matches.size());
    for (int i = 0; i < static_cast<int>(good_matches.size()); i++) {
//        printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i,
//                            good_matches[i].queryIdx, good_matches[i].trainIdx);
        point1 = keypoints_1[good_matches[i].queryIdx].pt;
        point2 = keypoints_2[good_matches[i].trainIdx].pt;

        // exclude duplicates
        if (std::find(right_features->begin(), right_features->end(), point2)
                != right_features->end()) {
            found_features--;
        } else {
            left_features->push_back(point1);
            right_features->push_back(point2);
        }
    }

    // skip displaying
    return found_features;

    while (1) {
        //-- Show detected matches
        cv::imshow("Good Matches", img_matches);
        if ((cv::waitKey(1) & 0xFF) == 27) {
            cv::destroyWindow("Good Matches");
            std::cout << "ESC key pressed by user" << std::endl;
            return found_features;
        }
    }
}

cv::Mat LoadParameters(std::string path, std::string mat) {
    cv::Mat calib_mat;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs[mat] >> calib_mat;
    return calib_mat;
}

double CalcDepth(double disp) {
//    return -8.4171e-05*pow(disp, 3) + 0.037*pow(disp,2) + -6.0613*disp + 478.7784;
    // new calib
//    return -3.0252e-04*pow(disp, 3) + 0.07982*pow(disp, 2) + -8.70497*disp + 531.93879;
//    return -2.1480e-04*pow(disp, 3) + 0.0625*pow(disp, 2) + -7.6419*disp + 507.9743;
    // limited range (350-200mm)
//    return -1.2909e-04*pow(disp, 3) + 0.0502*pow(disp, 2) + -7.2236*disp + 520.9408;
    // limited range tracking middle pt
//    return -1.3054e-04*pow(disp, 3) + 0.0499*pow(disp, 2) + -7.1550*disp + 514.1538;
    // limited range tracking with opencv calib rectification
    return -1.7775e-04*pow(disp, 3) + 0.0437*pow(disp, 2) + -4.7771*disp + 350.4368;
}


}  // namespace mc
