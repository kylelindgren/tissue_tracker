#include "ssim.hpp"

#define C1 static_cast<float>(0.01 * 255 * 0.01  * 255)
#define C2 static_cast<float>(0.03 * 255 * 0.03  * 255)

namespace ssim {

// sigma on block_size
double sigma(cv::Mat const &m, int i, int j, int block_size) {
    double sd = 0;

    cv::Mat m_tmp = m(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
    cv::Mat m_squared(block_size, block_size, CV_64F);

    multiply(m_tmp, m_tmp, m_squared);

    // E(x)
    double avg = mean(m_tmp)[0];
    // E(x²)
    double avg_2 = mean(m_squared)[0];

    sd = sqrt(avg_2 - avg * avg);

    return sd;
}

// Covariance
double cov(cv::Mat const &m1, cv::Mat const &m2, int i, int j, int block_size) {
    cv::Mat m3 = cv::Mat::zeros(block_size, block_size, m1.depth());
    cv::Mat m1_tmp = m1(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
    cv::Mat m2_tmp = m2(cv::Range(i, i + block_size), cv::Range(j, j + block_size));

    multiply(m1_tmp, m2_tmp, m3);

    double avg_ro = mean(m3)[0];      // E(XY)
    double avg_r  = mean(m1_tmp)[0];  // E(X)
    double avg_o  = mean(m2_tmp)[0];  // E(Y)

    double sd_ro = avg_ro - avg_o * avg_r;  // E(XY) - E(X)E(Y)

    return sd_ro;
}

// Mean squared error
double eqm(cv::Mat const &img1, cv::Mat const &img2) {
    int i, j;
    double eqm = 0;
    int height = img1.rows;
    int width = img1.cols;

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
            eqm += (img1.at<double>(i, j) - img2.at<double>(i, j)) *
                    (img1.at<double>(i, j) - img2.at<double>(i, j));

    eqm /= height * width;

    return eqm;
}


/**
     *	Compute the PSNR between 2 images
     */
double psnr(cv::Mat const &img_src, cv::Mat const &img_compressed) {
    int D = 255;
    return (10 * log10((D*D)/eqm(img_src, img_compressed)));
}


/**
     * Compute the SSIM between 2 images
     */
double ssim(cv::Mat const &img_src, cv::Mat const &img_compressed,
            int block_size, bool show_progress) {
    double ssim = 0;

    int nbBlockPerHeight = img_src.rows / block_size;
    int nbBlockPerWidth  = img_src.cols / block_size;

    for (int k = 0; k < nbBlockPerHeight; k++) {
        for (int l = 0; l < nbBlockPerWidth; l++) {
            int m = k * block_size;
            int n = l * block_size;

            double avg_o    = mean(img_src(cv::Range(k, k + block_size),
                                           cv::Range(l, l + block_size)))[0];
            double avg_r    = mean(img_compressed(cv::Range(k, k + block_size),
                                                  cv::Range(l, l + block_size)))[0];
            double sigma_o  = sigma(img_src, m, n, block_size);
            double sigma_r  = sigma(img_compressed, m, n, block_size);
            double sigma_ro = cov(img_src, img_compressed, m, n, block_size);

            ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) /
                    ((avg_o * avg_o + avg_r * avg_r + C1) *
                     (sigma_o * sigma_o + sigma_r * sigma_r + C2));
        }
        // Progress
        if (show_progress)
            std::cout << "\r>>SSIM [" << static_cast<int>(((static_cast<double>(k))
                                                      / nbBlockPerHeight) * 100) << "%]";
    }
    ssim /= nbBlockPerHeight * nbBlockPerWidth;

    if (show_progress) {
        std::cout << "\r>>SSIM [100%]" << std::endl;
        std::cout << "SSIM : " << ssim << std::endl;
    }

    return ssim;
}

float compute_quality_metrics_L(const cv::Mat ref, const cv::Mat im,
                              int block_size, double update_delay) {
    // average inits
    static double ssim_total = 0;
    static double count = 0;
    static double interval = 0;
    static double ave = 0;

    static struct timeval tv;
    static int64 usec;
    static int64 temp = 0;

    cv::Mat im_src;
    cv::Mat im_warped;

    // Loading pictures
    ref.copyTo(im_src);
    im.copyTo(im_warped);

    im_src.convertTo(im_src, CV_64F);
    im_warped.convertTo(im_warped, CV_64F);

    int height_o = im_src.rows;
    int height_r = im_warped.rows;
    int width_o  = im_src.cols;
    int width_r  = im_warped.cols;

    // Check pictures size
    if (height_o != height_r || width_o != width_r) {
        std::cout << "Images must have the same dimensions" << std::endl;
        return -1;
    }

    // Check if the block size is a multiple of height / width
    if (height_o % block_size != 0 || width_o % block_size != 0) {
        std::cout << "WARNING : Image WIDTH and HEIGHT should be divisible by "
                     "BLOCK_SIZE for the maximum accuracy" << std::endl
                << "HEIGHT : "     << height_o   << std::endl
                << "WIDTH : "      << width_o    << std::endl
                << "BLOCK_SIZE : " << block_size << std::endl
                << std::endl;
    }

    double ssim_val = ssim(im_src, im_warped, block_size);
//    double psnr_val = psnr(im_src, im_warped);

    // ave update
    gettimeofday(&tv, NULL);
    usec = tv.tv_usec;

    if (!temp)
        temp = usec;
    if (usec < temp) {
        usec += 1e6;
        interval += usec - temp;
        usec -= 1e6;
    } else {
        interval += usec - temp;
    }

    temp = usec;

    ssim_total += ssim_val;
    count++;

    if (interval*1e-6 >= update_delay) {
        ave = ssim_total / count;
        interval = count = ssim_total = 0;
    }

//    std::cout << "SSIM : " << ave << std::endl;
//    std::cout << "PSNR : " << psnr_val << std::endl;

    return static_cast<float>(ave);
}


float compute_quality_metrics_R(const cv::Mat ref, const cv::Mat im,
                              int block_size, double update_delay) {
    // average inits
    static double ssim_total = 0;
    static double count = 0;
    static double interval = 0;
    static double ave = 0;

    static struct timeval tv;
    static int64 usec;
    static int64 temp = 0;

    cv::Mat im_src;
    cv::Mat im_warped;

    // Loading pictures
    ref.copyTo(im_src);
    im.copyTo(im_warped);

    im_src.convertTo(im_src, CV_64F);
    im_warped.convertTo(im_warped, CV_64F);

    int height_o = im_src.rows;
    int height_r = im_warped.rows;
    int width_o  = im_src.cols;
    int width_r  = im_warped.cols;

    // Check pictures size
    if (height_o != height_r || width_o != width_r) {
        std::cout << "Images must have the same dimensions" << std::endl;
        return -1;
    }

    // Check if the block size is a multiple of height / width
    if (height_o % block_size != 0 || width_o % block_size != 0) {
        std::cout << "WARNING : Image WIDTH and HEIGHT should be divisible by "
                     "BLOCK_SIZE for the maximum accuracy" << std::endl
                << "HEIGHT : "     << height_o   << std::endl
                << "WIDTH : "      << width_o    << std::endl
                << "BLOCK_SIZE : " << block_size << std::endl
                << std::endl;
    }

    double ssim_val = ssim(im_src, im_warped, block_size);
//    double psnr_val = psnr(im_src, im_warped);

    // ave update
    gettimeofday(&tv, NULL);
    usec = tv.tv_usec;

    if (!temp)
        temp = usec;
    if (usec < temp) {
        usec += 1e6;
        interval += usec - temp;
        usec -= 1e6;
    } else {
        interval += usec - temp;
    }

    temp = usec;

    ssim_total += ssim_val;
    count++;

    if (interval*1e-6 >= update_delay) {
        ave = ssim_total / count;
        interval = count = ssim_total = 0;
    }

//    std::cout << "SSIM : " << ave << std::endl;
//    std::cout << "PSNR : " << psnr_val << std::endl;

    return static_cast<float>(ave);
}

}  // namespace ssim
