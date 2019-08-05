#ifndef __COMMMON_PROCESSING_UNIT_H__
#define __COMMMON_PROCESSING_UNIT_H__




#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/eigen.h>
#include <pcl/common/gaussian.h>
#include <pcl/common/transforms.h>



#include <pcl/io/io.h>  
#include <pcl/io/pcd_io.h> 
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>



using namespace std;
using namespace cv;
using namespace pcl;

//extern visualization::CloudViewer viewer;
//extern pcl::visualization::PCLVisualizer viewer1;



namespace CommonProcessingUnit
{
    
    void drawDisparity(const cv::Mat&  disparity, const double& duration, const int& disp_size);

    void disparityToDepth(const cv::Mat& disparity, cv::Mat& depth);
    int saveCloudPoint(const cv::Mat& depth);
    void viewerOneOff(visualization::PCLVisualizer& viewer);
    void viewerCoordinate(visualization::PCLVisualizer& viewer);
    int processPointCloud(const cv::Mat& depth);
    void preprocessingDisparity(const cv::Mat&  disparity, cv::Mat& result_filter);

    void drawColorDisparity(const cv::Mat&  disparity, const int& disp_size);

    void createRemapMat(cv::Mat& lmapx, cv::Mat& lmapy, cv::Mat& rmapx, cv::Mat& rmapy, cv::Mat referenceimage);
    void rectifyStereo(const Mat& leftImg, const Mat& rightImg, Mat& leftR, Mat& rightR, const cv::Mat& lmapx, const cv::Mat& lmapy, const cv::Mat& rmapx, const cv::Mat& rmapy);

    void sameScaleConversion(const cv::Mat& depth);
    struct ImagePair
    {
        cv::Mat imgL;
        cv::Mat imgR;
        int imgID;

    };

    template<typename PointT> void
    savePLY(pcl::PointCloud<PointT> &cloud, const string& strPly);


}

#endif // DEBUG