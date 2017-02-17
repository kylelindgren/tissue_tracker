#ifndef MC_HPP
#define MC_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <ctime>
#include <queue>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include <armadillo>

#define CP_NUM   4
#define IMWIDTH  640
#define IMHEIGHT 480
#define ROI_W    80
#define ROI_H    80
#define PIXELS   ROI_W*ROI_H
#define KSIZE    1  // kernel size for sobel filtering

#define BASELINE 9   // in mm
#define FOCAL    18.5636  // in mm from cv::calibrationMatrixValues() opencv_calib.yaml


namespace mc {

double TPS(double pt1x, double pt1y, double pt2x, double pt2y);
void CalcMXT(const cv::Mat h_0_L, const cv::Mat h_0_R, const cv::Mat frame_0_L,
             const cv::Mat frame_0_R, const cv::Mat X_L, cv::Mat *X_R, cv::Mat *M_L,
             cv::Mat *M_R, cv::Mat *T_L, cv::Mat *T_R);
void CalcK(const cv::Mat h, cv::Mat *K);
void UpdateDepth(float focal, const cv::Mat h_a_L, const cv::Mat h_a_R, cv::Mat *h_depth);
void OnMouse(int event, int x, int y, int, void* center);
void DrawROIBorder(const cv::Mat MK_L, const cv::Mat MK_R,
                   const cv::Mat h, bool left, cv::Mat *frame);
void DrawInitBorder(int roi_0x_L, int roi_0y_L, cv::Mat *frame);
int ComputeJointHistogram(int n_bins, int size_bins, float expected_L[], float expected_R[],
                          float p_joint_L[], float p_joint_R[], const cv::Mat frame_L,
                          const cv::Mat frame_R, const cv::Mat frame_0_L, const cv::Mat frame_0_R);
void ResetExpected(int n_bins, float expected_L[], float expected_R[]);
void ComputeExpectedImg(const cv::Mat frame_0_L, const cv::Mat frame_0_R, float correction_L[],
                        float correction_R[], float expected_L[], float expected_R[], int n_bins,
                        int size_bins, cv::Mat *frame_comp_L, cv::Mat *frame_comp_R,
                        cv::Mat *gradx_comp_L, cv::Mat *gradx_comp_R,
                        cv::Mat *grady_comp_L, cv::Mat *grady_comp_R);
void UpdateJ_0(const cv::Mat MK_L, const cv::Mat MK_R, const cv::Mat X_L, const cv::Mat X_R,
               const cv::Mat gradx_comp_L, const cv::Mat gradx_comp_R,
               const cv::Mat grady_comp_L, const cv::Mat grady_comp_R,
               cv::Mat *J_0_L, cv::Mat *J_0_R);
void Jacobian(const cv::Mat I, const cv::Mat MK, cv::Mat *J);
void DrawGrid(const cv::Mat MK_L, const cv::Mat h, cv::Mat *image);
void AffineTrans(const cv::Mat MK_L, const cv::Mat MK_R, const cv::Mat X_L,
                 const cv::Mat X_R, const cv::Mat h_a, bool left, cv::Mat *image);
int MatchFeatures(const cv::Mat left, const cv::Mat right, std::vector<cv::Point2f> *left_features,
                  std::vector<cv::Point2f> *right_features);
cv::Mat LoadParameters(std::string path, std::string mat);
double CalcDepth(double disp);

}  // namespace mc

#endif  // MC_HPP
