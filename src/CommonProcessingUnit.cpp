#include "../include/CommonProcessingUnit.h"
#include <pcl/io/ply_io.h>

using namespace cv;

int user_data;

namespace CommonProcessingUnit
{ 
    template <class... Args>
    static std::string format_string(const char* fmt, Args... args)
    {
        const int BUF_SIZE = 1024;

        char buf[BUF_SIZE];
        std::snprintf(buf, BUF_SIZE, fmt, args...);
        return std::string(buf);
    }

    void createRemapMat(cv::Mat& lmapx, cv::Mat& lmapy, cv::Mat& rmapx, cv::Mat& rmapy, cv::Mat referenceimage)
    {
        Mat R1, R2, P1, P2, Q;
        Mat K1, K2, R;
        Vec3d T;
        Mat D1, D2;
        char* calib_file = "../stereo_parameter/cam_stereo.yml";

        cv::FileStorage fs1(calib_file, cv::FileStorage::READ);
        fs1["K1"] >> K1;
        fs1["K2"] >> K2;
        fs1["D1"] >> D1;
        fs1["D2"] >> D2;
        fs1["R"] >> R;
        fs1["T"] >> T;

        fs1["R1"] >> R1;
        fs1["R2"] >> R2;
        fs1["P1"] >> P1;
        fs1["P2"] >> P2;
        fs1["Q"] >> Q;

        
        cv::Mat imgU1, imgU2;

        cv::initUndistortRectifyMap(K1, D1, R1, P1, referenceimage.size(), CV_32F, lmapx, lmapy);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, referenceimage.size(), CV_32F, rmapx, rmapy);

    }

    void rectifyStereo(const Mat& leftImg, const Mat& rightImg, Mat& leftR ,Mat& rightR,
        const cv::Mat& lmapx, const cv::Mat& lmapy, const cv::Mat& rmapx, const cv::Mat& rmapy)
    {

        /*Mat R1, R2, P1, P2, Q;
        Mat K1, K2, R;
        Vec3d T;
        Mat D1, D2;
        char* calib_file = "../stereo_parameter/cam_stereo.yml";

        cv::FileStorage fs1(calib_file, cv::FileStorage::READ);
        fs1["K1"] >> K1;
        fs1["K2"] >> K2;
        fs1["D1"] >> D1;
        fs1["D2"] >> D2;
        fs1["R"] >> R;
        fs1["T"] >> T;

        fs1["R1"] >> R1;
        fs1["R2"] >> R2;
        fs1["P1"] >> P1;
        fs1["P2"] >> P2;
        fs1["Q"] >> Q;

        cv::Mat lmapx, lmapy, rmapx, rmapy;
        cv::Mat imgU1, imgU2;
        
        cv::initUndistortRectifyMap(K1, D1, R1, P1, leftImg.size(), CV_32F, lmapx, lmapy);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, rightImg.size(), CV_32F, rmapx, rmapy);*/
        
        cv::remap(leftImg, leftR, lmapx, lmapy, cv::INTER_LINEAR);
        cv::remap(rightImg, rightR, rmapx, rmapy, cv::INTER_LINEAR);

    }

    void drawDisparity(const cv::Mat&  disparity, const double& duration, const int& disp_size)
    {

        // draw results
        cv::Mat disparity_8u, disparity_color;
        imwrite("../image/disparity.png", disparity);
 
        disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);

        cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);

        imwrite("../image/disparity_color.png", disparity_color);

        const double fps = 1e3 / duration;
        cv::putText(disparity_color, format_string("sgm execution time: %4.1f[msec] %4.1f[FPS]", duration, fps),

        cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

        cv::imshow("disparity", disparity_color);

    }

    void preprocessingDisparity(const cv::Mat&  disparity, cv::Mat& result_filter)
    {

       ///* cv::medianBlur(disparity, result_filter, 7);*/
       // const int d = 6;
       // const float sigmaColor = 20;
       // const float sigmaSpace = 20;

       // // 高斯双边滤波
       //Mat result_bilateral;
       //cv::bilateralFilter(disparity, result_bilateral, d, sigmaColor, sigmaSpace);
       ////result_filter = disparity;
       //// cv::imshow("disparity1", result_filter);



       // Mat thresh;

       // threshold(result_bilateral, thresh, 1, 255, 0);

       // //imshow("thresh", thresh);


       // Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));	//开运算
       // Mat open_result;	
       // morphologyEx(thresh, open_result, MORPH_OPEN,element);	

       // //imshow("开运算", open_result);
       // Mat close_result;	
       // morphologyEx(open_result, close_result, MORPH_CLOSE,element);
       // //imshow("闭运算", close_result);
       // 
       // Mat result(disparity.rows, disparity.cols, CV_8UC1);

       // for (int j = 0; j < disparity.rows; ++j)
       // {
       //     for (int i = 0; i < disparity.cols; ++i)
       //     {
       //         if (open_result.at<uchar>(j, i) > 0)
       //             result.at<uchar>(j, i) = disparity.at<uchar>(j, i);
       //        
       //     }
       // }
       //
       // result_filter = result;


        

        cv::medianBlur(disparity, result_filter, 3);
        
      
      
    }

    void disparityToDepth(const cv::Mat& imgDisparity8U, cv::Mat& depth)
    {
        //Depth
        Mat Q;
        FileStorage fs("../stereo_parameter/cam_stereo.yml", FileStorage::READ);
        fs["Q"] >> Q;
        fs.release();
        reprojectImageTo3D(imgDisparity8U, depth, Q, false, -1);
    }

    void viewerOneOff(visualization::PCLVisualizer& viewer)
    {
        viewer.setBackgroundColor(0.0, 0.0, 0.0);
        //viewer.addCoordinateSystem();
        
    }

    void viewerCoordinate(visualization::PCLVisualizer& viewer)
    {
        viewer.addCoordinateSystem();

    }

    int saveCloudPoint(const cv::Mat& depth)
    {
       

       if (depth.empty())
       {
           cout << "The image is empty, please check it!" << endl;
           return -1;
       }

       // Point cloud under camera coordinate system
        PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
        for (int j = 0; j < depth.rows; ++j)
        {
            for (int i = 0; i < depth.cols; ++i)
            {
                auto d = depth.at<cv::Vec3f>(j, i)[2];
                 if (d == INFINITY || d == 0 || d > 5)
                    continue;
                PointXYZ p;
                p.z = -d*1000; // Zc = baseline * f / (d + doffs)
                p.x = depth.at<Vec3f>(j, i)[0] * 1000; //XC to the right, Yc down as positive
                p.y = -depth.at<Vec3f>(j, i)[1] * 1000;

               
                cloud->points.push_back(p);
            }
        }

        cloud->height = depth.rows;
        cloud->width = depth.cols;
        cloud->points.resize(cloud->height * cloud->width);
        
       
       
        pcl::io::savePLYFile("../image/cloud.ply", *cloud);
       
        
        return 0;
    }

    int processPointCloud(const cv::Mat& depth)
    {
        int distanceFactor = 1000;
        
        if (depth.empty())
        {
            cout << "The image is empty, please check it!" << endl;
            return -1;
        }

        // Point cloud under camera coordinate system
        PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
        for (int j = 0; j < depth.rows; ++j)
        {
            for (int i = 0; i < depth.cols; ++i)
            {
                auto d = depth.at<cv::Vec3f>(j, i)[2];
                if (d == INFINITY || d == 0 || d > 5)
                    continue;
                PointXYZ p;
                p.z = -d * distanceFactor; // Zc = baseline * f / (d + doffs)
                p.x = depth.at<Vec3f>(j, i)[0] * distanceFactor; //XC to the right, Yc down as positive
                p.y = -depth.at<Vec3f>(j, i)[1] * distanceFactor;
                cloud->points.push_back(p);
            }
        }

        cloud->height = depth.rows;
        cloud->width = depth.cols;
        cloud->points.resize(cloud->height * cloud->width);
       
       

        return 0;
    }

    void drawColorDisparity(const cv::Mat&  disparity, const int& disp_size)
    {
        // draw results
        cv::Mat disparity_8u, disparity_color;
        disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
        cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);

        cv::imshow("disparity_color", disparity_color);
    }


    void sameScaleConversion(const cv::Mat& depth)
    {
        vector<cv::Mat> channels;
        cv::split(depth, channels);
        cv::Mat test = channels.at(2);
        cv::Mat matScale;
        test.convertTo(matScale, CV_8UC1, 255 / 5);
        // std::cout << depth_mat.channels() << std::endl;
        imshow("matScaleGray", matScale);
        cv::Mat imColor;
        cv::applyColorMap(matScale, imColor, cv::COLORMAP_JET);
        imshow("imColor", imColor);
    
    }








}