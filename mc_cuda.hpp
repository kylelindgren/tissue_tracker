#ifndef MC_CUDA_HPP
#define MC_CUDA_HPP

#include "mc.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"


namespace mc_cuda {

//double TPS(double pt1x, double pt1y, double pt2x, double pt2y);
//void UpdateDepth(float focal, const cv::Mat h_a_L,
//                 const cv::Mat h_a_R, cv::Mat *h_depth);
//void OnMouse(int event, int x, int y, int, void* center);
//int ComputeJointHistogram(int n_bins, int size_bins, float expected_L[], float expected_R[],
//                          float p_joint_L[], float p_joint_R[], const cv::cuda::GpuMat frame_L,
//                          const cv::cuda::GpuMat frame_R, const cv::cuda::GpuMat frame_0_L,
//                          const cv::cuda::GpuMat frame_0_R);
//void ResetExpected(int n_bins, float expected_L[], float expected_R[]);
void ComputeExpectedImg(const cv::Mat frame_0_L, const cv::Mat frame_0_R,
                        float correction_L[], float correction_R[], float expected_L[],
                        float expected_R[], int n_bins, int size_bins,
                        cv::Ptr<cv::cuda::Filter> filter_x, cv::Ptr<cv::cuda::Filter> filter_y,
                        cv::cuda::GpuMat *frame_comp_L, cv::cuda::GpuMat *frame_comp_R,
                        cv::cuda::GpuMat *gradx_comp_L, cv::cuda::GpuMat *gradx_comp_R,
                        cv::cuda::GpuMat *grady_comp_L, cv::cuda::GpuMat *grady_comp_R);
void Jacobian(const cv::Mat I, cv::Ptr<cv::cuda::Filter> sobel_x,
              cv::Ptr<cv::cuda::Filter> sobel_y, const cv::Mat MK, cv::Mat *J);
//void DrawGrid(const cv::cuda::GpuMat MK_L, const cv::cuda::GpuMat h, cv::cuda::GpuMat *image);
void AffineTrans(const cv::Mat MK_L, const cv::Mat MK_R,
                 const cv::Mat X_L, const cv::Mat X_R,
                 const cv::Mat h_a, bool left, cv::Mat *image);
//double CalcDepth(double disp);

}  // namespace mc_cuda

#endif  // MC_CUDA_HPP
