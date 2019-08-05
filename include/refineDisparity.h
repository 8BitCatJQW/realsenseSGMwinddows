#ifndef __COMMMON_REFINEDIPARITY_UNIT_H__
#define __COMMMON_REFINEDIPARITY_UNIT_H__

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>


namespace RefineDisparity
{
	void adaptiveMeanFilter(cv::Mat & D);
	void medianFilter(cv::Mat & D);

}

#endif // DEBUG