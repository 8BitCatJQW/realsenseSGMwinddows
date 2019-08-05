#include "GxIAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include<thread>
#include<chrono>
#include<memory>
#include <mutex>
#include <queue>


#include "include/CommonProcessingUnit.h"
#include "include/common.h"
#include "include/refineDisparity.h"
#include "include/guidedfilter.h"

#include <librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_processing.hpp>
#include <librealsense2/hpp/rs_types.hpp>
#include <librealsense2/hpp/rs_sensor.hpp>

using namespace pcl;
using namespace std;
using namespace cv;

using namespace CommonProcessingUnit;
using namespace StereoMatching;
using namespace RefineDisparity;

//pcl::visualization::PCLVisualizer viewer1("viewer_1");

queue<ImagePair> pairs;
mutex mu;  //线程互斥对象  

bool working = true;

float cropFactor = 0.5;

cv::Mat lmapx, lmapy, rmapx, rmapy;




int main()
{

   
    int count = 0;
    std::string num;
    rs2::pipeline pipe;     //Contruct a pipeline which abstracts the device
    rs2::config cfg;    //Create a configuration for configuring the pipeline with a non default profile
    int w = 1280;
    int h = 720;
    cfg.enable_stream(RS2_STREAM_COLOR, w, h, RS2_FORMAT_BGR8, 15);
    cfg.enable_stream(RS2_STREAM_DEPTH, w, h, RS2_FORMAT_Z16, 15);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, w, h, RS2_FORMAT_Y8, 15);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, w, h, RS2_FORMAT_Y8, 15);
    rs2::pipeline_profile selection = pipe.start(cfg);
    rs2::colorizer color_map;


    rs2::frameset frames;
    frames = pipe.wait_for_frames();

    //Get each frame
   
   
    auto IR_frame_left = frames.get_infrared_frame(1);
    auto IR_frame_right = frames.get_infrared_frame(2);

  

    cv::Mat I1(cv::Size(w, h), CV_8UC1, (void*)IR_frame_left.get_data());
    cv::Mat I2(cv::Size(w, h), CV_8UC1, (void*)IR_frame_right.get_data());
   

  

    //-----------------------SGM Initialization-------------------------------
   
    int disp_size = 128;

  
    
   
    //II1.convertTo(I1, CV_8U);
    //II2.convertTo(I2, CV_8U);
   /* imshow("test", I1);
    std::cout << "sss" << I1.type() << std::endl;
    waitKey(0);*/
  
    ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
    ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
    ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
    ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");

    int width = I1.cols;
    int height = I1.rows;
    cout << I1.type() << endl;

    const int input_depth = I1.type() == CV_8U ? 8 : 16;
    const int input_bytes = input_depth * width * height / 8;
    const int output_depth = 8;
    const int output_bytes = output_depth * width * height / 8;

    sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
    device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);
    cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U);

    
   
    while (1)
    {
   
              frames = pipe.wait_for_frames();

               IR_frame_left = frames.get_infrared_frame(1);
               IR_frame_right = frames.get_infrared_frame(2);

              cv::Mat II1(cv::Size(w, h), CV_8UC1, (void*)IR_frame_left.get_data());
              cv::Mat II2(cv::Size(w, h), CV_8UC1, (void*)IR_frame_right.get_data());


                //imwrite("../image/left.png", imgs.imgL);
                //imwrite("../image/right.png", imgs.imgR);
               // cv::waitKey(1);
                //rectifyStereo(imgs.imgL, imgs.imgR, leftR, rightR, lmapx, lmapy, rmapx, rmapy);
              
                cv::Mat leftR = II1.clone();
                cv::Mat rightR = II2.clone();

                cv::equalizeHist(leftR, leftR);

                cv::equalizeHist(rightR, rightR);

                cudaMemcpy(d_I1.data, leftR.data, input_bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(d_I2.data, rightR.data, input_bytes, cudaMemcpyHostToDevice);
                sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
                cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
                if (disparity.empty())
                {
                    cout << "data empty" << endl;
                    break;
                }
                Mat result;
               

               
                drawColorDisparity(disparity,  128);
                
               

            int flag = cv::waitKey(1);
            if (flag == 27)
            {
                working = false;
                break;
            }
            if (flag == 32)
            {
                cv::imwrite("../image/left.png", leftR);
                cv::imwrite("../image/right.png", rightR);
                cv::imwrite("../image/disparity.png", disparity);

                cv::Mat depth;
                //cv::Mat d = cv::imread("../image/3.png",0);
                disparityToDepth(disparity, depth);
                saveCloudPoint(depth);
               
            }

         
     }

    

   

    return 0;
}

