#ifndef KINECT_INTEGRATION_H
#define KINECT_INTEGRATION_H

#include "opencv2/opencv.hpp"
#include <Windows.h>
#include <Kinect.h>

class KinectIntegration {
public:
	KinectIntegration();
	~KinectIntegration();
	cv::Point drawEllipse(cv::Mat& image, const Joint& joint, const int radius, const cv::Vec3b& color, const int thickness);
	int kinectLoop();

private:
	template<class Interface>
	void SafeRelease(Interface *& pInterfaceToRelease);

	IKinectSensor* pSensor;
	IColorFrameSource* pColorSource;
	IColorFrameReader* pColorReader;
	IFrameDescription* pColorDescription;
	IDepthFrameSource* pDepthSource;
	IDepthFrameReader* pDepthReader;
	IFrameDescription* pDescription;
	IBodyFrame* pBodyFrame;
	IBodyFrameReader* pBodyFrameReader;
	IBodyFrameSource* pBodyFrameSource;
	ICoordinateMapper* mapper;         // Converts between depth, color, and 3d coordinates
};

#endif