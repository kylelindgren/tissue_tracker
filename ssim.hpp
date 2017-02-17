#ifndef SSIM_HPP
#define SSIM_HPP
// getting time
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace ssim {

double sigma(cv::Mat const &m, int i, int j, int block_size);
double cov(cv::Mat const &m1, cv::Mat const &m2, int i, int j, int block_size);
double eqm(cv::Mat const &img1, cv::Mat const &img2);
double psnr(cv::Mat const &img_src, cv::Mat const &img_compressed);
double ssim(cv::Mat const &img_src, cv::Mat const &img_compressed,
            int block_size, bool show_progress=false);
// two compute_quality_metrics functions to keep static function members separate
float compute_quality_metrics_L(cv::Mat const ref, cv::Mat const im,
                              int block_size, double update_delay=0);
float compute_quality_metrics_R(cv::Mat const ref, cv::Mat const im,
                              int block_size, double update_delay=0);

}  // namespace ssim

#endif  // SSIM_HPP
