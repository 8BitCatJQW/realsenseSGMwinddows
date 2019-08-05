#include "../include/refineDisparity.h"


namespace RefineDisparity
{ 

    void adaptiveMeanFilter(cv::Mat & D)
    {
        // get disparity image dimensions
        int32_t D_width = D.cols;
        int32_t D_height = D.rows;

        // allocate temporary memory
        cv::Mat D_copy = D.clone();
        cv::Mat D_tmp(D.rows, D.cols, CV_32FC1, cv::Scalar::all(0));

        // zero input disparity maps to -10 (this makes the bilateral
        // weights of all valid disparities to 0 in this region)
        for (int32_t u = 0; u < D_width; u++) {
            for (int32_t v = 0; v < D_height; v++) {
                if (D.at<float>(v, u) < 0) {
                    D_copy.at<float>(v, u) = -10;
                    D_tmp.at<float>(v, u) = -10;
                }
            }
        }

        __m128 xconst0 = _mm_set1_ps(0);
        __m128 xconst4 = _mm_set1_ps(4);
        __m128 xval, xweight1, xweight2, xfactor1, xfactor2;

        float *val = (float *)_mm_malloc(8 * sizeof(float), 16);
        float *weight = (float*)_mm_malloc(4 * sizeof(float), 16);
        float *factor = (float*)_mm_malloc(4 * sizeof(float), 16);

        // set absolute mask
        __m128 xabsmask = _mm_set1_ps(0x7FFFFFFF);


        // horizontal filter
        for (int32_t v = 3; v < D_height - 3; v++) {

            // init
            for (int32_t u = 0; u < 7; u++)
                val[u] = D_copy.at<float>(v, u);

            // loop
            for (int32_t u = 7; u < D_width; u++) {

                // set
                float val_curr = D_copy.at<float>(v, u - 3);
                val[u % 8] = D_copy.at<float>(v, u);

                xval = _mm_load_ps(val);
                xweight1 = _mm_sub_ps(xval, _mm_set1_ps(val_curr));
                xweight1 = _mm_and_ps(xweight1, xabsmask);
                xweight1 = _mm_sub_ps(xconst4, xweight1);
                xweight1 = _mm_max_ps(xconst0, xweight1);
                xfactor1 = _mm_mul_ps(xval, xweight1);

                xval = _mm_load_ps(val + 4);
                xweight2 = _mm_sub_ps(xval, _mm_set1_ps(val_curr));
                xweight2 = _mm_and_ps(xweight2, xabsmask);
                xweight2 = _mm_sub_ps(xconst4, xweight2);
                xweight2 = _mm_max_ps(xconst0, xweight2);
                xfactor2 = _mm_mul_ps(xval, xweight2);

                xweight1 = _mm_add_ps(xweight1, xweight2);
                xfactor1 = _mm_add_ps(xfactor1, xfactor2);

                _mm_store_ps(weight, xweight1);
                _mm_store_ps(factor, xfactor1);

                float weight_sum = weight[0] + weight[1] + weight[2] + weight[3];
                float factor_sum = factor[0] + factor[1] + factor[2] + factor[3];

                if (weight_sum > 0) {
                    float d = factor_sum / weight_sum;
                    if (d >= 0) D_tmp.at<float>(v, u - 3) = d;
                }
            }
        }

        // vertical filter
        for (int32_t u = 3; u < D_width - 3; u++) {

            // init
            for (int32_t v = 0; v < 7; v++)
                val[v] = D_tmp.at<float>(v, u);

            // loop
            for (int32_t v = 7; v < D_height; v++) {

                // set
                float val_curr = D_tmp.at<float>(v - 3, u);
                val[v % 8] = D_tmp.at<float>(v, u);

                xval = _mm_load_ps(val);
                xweight1 = _mm_sub_ps(xval, _mm_set1_ps(val_curr));
                xweight1 = _mm_and_ps(xweight1, xabsmask);
                xweight1 = _mm_sub_ps(xconst4, xweight1);
                xweight1 = _mm_max_ps(xconst0, xweight1);
                xfactor1 = _mm_mul_ps(xval, xweight1);

                xval = _mm_load_ps(val + 4);
                xweight2 = _mm_sub_ps(xval, _mm_set1_ps(val_curr));
                xweight2 = _mm_and_ps(xweight2, xabsmask);
                xweight2 = _mm_sub_ps(xconst4, xweight2);
                xweight2 = _mm_max_ps(xconst0, xweight2);
                xfactor2 = _mm_mul_ps(xval, xweight2);

                xweight1 = _mm_add_ps(xweight1, xweight2);
                xfactor1 = _mm_add_ps(xfactor1, xfactor2);

                _mm_store_ps(weight, xweight1);
                _mm_store_ps(factor, xfactor1);

                float weight_sum = weight[0] + weight[1] + weight[2] + weight[3];
                float factor_sum = factor[0] + factor[1] + factor[2] + factor[3];

                if (weight_sum > 0) {
                    float d = factor_sum / weight_sum;
                    if (d >= 0) D.at<float>(v - 3, u);
                }
            }
        }


        // free memory
        _mm_free(val);
        _mm_free(weight);
        _mm_free(factor);
    }

    /**
     * Median filter
     */
    void medianFilter(cv::Mat & D)
    {
        // get disparity image dimensions
        int32_t D_width = D.cols;
        int32_t D_height = D.rows;

        // temporary memory
        cv::Mat D_temp(D.rows, D.cols, CV_32FC1, cv::Scalar::all(0));
        int32_t window_size = 3;

        float *vals = new float[window_size * 2 + 1];
        int32_t i, j;
        float temp;

        // first step: horizontal median filter
        for (int32_t u = window_size; u<D_width - window_size; u++) {
            for (int32_t v = window_size; v<D_height - window_size; v++) {
                if (D.at<float>(v, u) >= 0) {
                    j = 0;
                    for (int32_t u2 = u - window_size; u2 <= u + window_size; u2++) {
                        temp = D.at<float>(v, u2);
                        i = j - 1;
                        while (i >= 0 && *(vals + i)>temp) {
                            *(vals + i + 1) = *(vals + i);
                            i--;
                        }
                        *(vals + i + 1) = temp;
                        j++;
                    }
                    D_temp.at<float>(v, u) = *(vals + window_size);
                }
                else {
                    D_temp.at<float>(v, u) = D.at<float>(v, u);
                }

            }
        }

        // second step: vertical median filter
        for (int32_t u = window_size; u<D_width - window_size; u++) {
            for (int32_t v = window_size; v<D_height - window_size; v++) {
                if (D.at<float>(v, u) >= 0) {
                    j = 0;
                    for (int32_t v2 = v - window_size; v2 <= v + window_size; v2++) {
                        temp = D_temp.at<float>(v2, u);
                        i = j - 1;
                        while (i >= 0 && *(vals + i)>temp) {
                            *(vals + i + 1) = *(vals + i);
                            i--;
                        }
                        *(vals + i + 1) = temp;
                        j++;
                    }
                    D.at<float>(v, u) = *(vals + window_size);
                }
                else {
                    D.at<float>(v, u) = D.at<float>(v, u);
                }
            }
        }

        delete[] vals;
    }

}
