#include "../header/kinectIntegration.h"
#include <thread>
#include <ppl.h>

#include <stdlib.h>
#include <sstream>
#include <experimental/filesystem>
#include <string>
#include <fstream>
#include <cerrno>
#include <iostream>
#include <stdint.h>
#include <ctype.h>
#include <chrono>

int SAVE_INT = 1;
//Joint joints[JointType_Count];              // List of joints in the tracked body

template<class Interface>
inline void KinectIntegration::SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL) {
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

KinectIntegration::KinectIntegration()
{
	cv::setUseOptimized(true);

	// Sensor
	HRESULT hResult = S_OK;
	hResult = GetDefaultKinectSensor(&pSensor);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : GetDefaultKinectSensor");
	}

	hResult = pSensor->Open();
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : IKinectSensor::Open()");
	}

	//--------------------[Color]---------------------------
	// Source
	hResult = pSensor->get_ColorFrameSource(&pColorSource);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : IKinectSensor::get_ColorFrameSource()");
	}

	// Reader
	hResult = pColorSource->OpenReader(&pColorReader);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : IColorFrameSource::OpenReader()");
	}

	// Description
	hResult = pColorSource->get_FrameDescription(&pColorDescription);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : IColorFrameSource::get_FrameDescription()");
	}

	//--------------------[Depth]---------------------------
	// Source
	hResult = pSensor->get_DepthFrameSource(&pDepthSource);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : IKinectSensor::get_DepthFrameSource()");
	}

	// Reader
	hResult = pDepthSource->OpenReader(&pDepthReader);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : IDepthFrameSource::OpenReader()");
	}

	// Description
	hResult = pDepthSource->get_FrameDescription(&pDescription);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : IDepthFrameSource::get_FrameDescription()");
	}

	//--------------------[Mapper]--------------------------
	if (pSensor) {
		pSensor->get_CoordinateMapper(&mapper);
	}
	else {
		throw std::invalid_argument("Error : get_CoordinateMapper()");
	}

	//--------------------[Body]-----------------------------
	hResult = pSensor->get_BodyFrameSource(&pBodyFrameSource);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : get_BodyFrameSource()");
	}

	hResult = pBodyFrameSource->OpenReader(&pBodyFrameReader);
	if (FAILED(hResult)) {
		throw std::invalid_argument("Error : IBodyFrameSource::OpenReader()");
	}

	SafeRelease(pBodyFrameSource);
}

KinectIntegration::~KinectIntegration()
{
	SafeRelease(pDepthSource);
	SafeRelease(pDepthReader);
	SafeRelease(pDescription);
	if (pSensor) {
		pSensor->Close();
	}
	SafeRelease(pSensor);
	cv::destroyAllWindows();
}


// Draw Ellipse
inline cv::Point KinectIntegration::drawEllipse(cv::Mat& image, const Joint& joint, const int radius, const cv::Vec3b& color, const int thickness)
{
	if (image.empty()) {
		return cv::Point(-1, -1);
	}

	// Convert Coordinate System and Draw Joint
	ColorSpacePoint colorSpacePoint;
	mapper->MapCameraPointToColorSpace(joint.Position, &colorSpacePoint);
	const int x = static_cast<int>(colorSpacePoint.X + 0.5f);
	const int y = static_cast<int>(colorSpacePoint.Y + 0.5f);
	if ((0 <= x) && (x < image.cols) && (0 <= y) && (y < image.rows)) {
		cv::circle(image, cv::Point(x, y), radius, static_cast<cv::Scalar>(color), thickness, 16);
	}

	return cv::Point(x, y);
}


int KinectIntegration::kinectLoop()
{
	HRESULT hResultColor, hResultDepth, hResultBody;
	//-------------------------[Color Initialization]--------------------------
	int cWidth = 0;
	int cHeight = 0;
	unsigned int colorBytesPerPixel;
	pColorDescription->get_Width(&cWidth); // 1920
	pColorDescription->get_Height(&cHeight); // 1080
	pColorDescription->get_BytesPerPixel(&colorBytesPerPixel);
	unsigned int colorBufferSize = cWidth * cHeight * 4 * sizeof(unsigned char);

	//-------------------------[Depth Initialization]--------------------------
	int dWidth = 0;
	int dHeight = 0;
	pDescription->get_Width(&dWidth); // 512
	pDescription->get_Height(&dHeight); // 424
	unsigned int bufferSize = dWidth * dHeight * sizeof(unsigned short);

	// Range ( Range of Depth is 500-8000[mm], Range of Detection is 500-4500[mm] ) 
	unsigned short min = 0;
	unsigned short max = 0;
	pDepthSource->get_DepthMinReliableDistance(&min); // 500
	pDepthSource->get_DepthMaxReliableDistance(&max); // 4500
	std::cout << "Range : " << min << " - " << max << std::endl;

	//-------------------------------------------------------------------------
	const int scalingFactor = 1;

	cv::Mat colorBufferMat(cHeight, cWidth, CV_8UC4);
	cv::Mat color(cHeight / scalingFactor, cWidth / scalingFactor, CV_8UC4);

	cv::Mat depth(dHeight, dWidth, CV_16UC1);

	//-------------------------------------------------------------------------
	cv::namedWindow("Joints", CV_WINDOW_NORMAL);
	//cv::namedWindow("Color Original", CV_WINDOW_NORMAL);
	cv::namedWindow("Color", CV_WINDOW_NORMAL);
	cv::namedWindow("Depth aligned", CV_WINDOW_NORMAL);
	cv::namedWindow("Depth");

	std::vector<BYTE> colorBuffer(cWidth * cHeight * 4);
	std::vector<UINT16> depthBuffer(dWidth * dHeight);

	std::vector<DepthSpacePoint> depthSpacePoints(cWidth * cHeight);
	std::vector<ColorSpacePoint> colorSpacePoints(dWidth * dHeight);

	std::vector<cv::Mat> colorCache;
	std::vector<cv::Mat> depthCache;

	//-------------------------------------------------------------------------
	bool toggleSave = false;
	int datasetCount = 0;
	int frameCount = 0;
	std::ofstream txtWriterDepth, txtWriterColor;
	std::string DATASET_NAME = "final";
	std::string COLOR_NAME = "rgb";
	std::string DEPTH_NAME = "depth";
	std::string PATH_NAME = "";


	std::vector<int> p;
	p.push_back(CV_IMWRITE_PNG_COMPRESSION);
	p.push_back(0); // compression factor

	int curFrameCount = 0;
	unsigned char key;
	bool key_edge = false;

	// File to store joints; initialize timestamps
	std::ofstream file, tfile;
	std::chrono::time_point<std::chrono::system_clock> estBegin, estEnd;
	estBegin = std::chrono::system_clock::now();

	while (1) {
		// Capture RGB-D frames from the KinectV2
		IColorFrame* pColorFrame = nullptr;
		IDepthFrame* pDepthFrame = nullptr;

		hResultColor = pColorReader->AcquireLatestFrame(&pColorFrame);
		hResultDepth = pDepthReader->AcquireLatestFrame(&pDepthFrame);
		hResultBody = pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);

		// Only process if we have received frames
		if (SUCCEEDED(hResultDepth) && SUCCEEDED(hResultColor)) {
			pColorFrame->CopyConvertedFrameDataToArray(colorBufferSize, reinterpret_cast<BYTE*>(colorBufferMat.data), ColorImageFormat::ColorImageFormat_Bgra);
			cv::resize(colorBufferMat, color, cv::Size(cWidth / scalingFactor, cHeight / scalingFactor));

			pDepthFrame->CopyFrameDataToArray(static_cast<UINT>(depthBuffer.size()), &depthBuffer[0]);
			cv::Mat depth = cv::Mat(dHeight, dWidth, CV_16UC1, &depthBuffer[0]).clone();

			// mapper->MapColorFrameToDepthSpace(depthBuffer.size(), &depthBuffer[0], depthSpacePoints.size(), &depthSpacePoints[0]);
			// Mapping Color to Depth Resolution
			// From https://github.com/UnaNancyOwen/Kinect2Sample
			// Retrieve Mapped 
			mapper->MapColorFrameToDepthSpace(depthBuffer.size(), &depthBuffer[0], depthSpacePoints.size(), &depthSpacePoints[0]);

			// Mapping Depth to Color Resolution
			std::vector<UINT16> buffer(cWidth * cHeight);

			Concurrency::parallel_for(0, cHeight, [&](const int colorY) {
				const unsigned int colorOffset = colorY * cWidth;
				for (int colorX = 0; colorX < cWidth; colorX++) {
					const unsigned int colorIndex = colorOffset + colorX;
					const int depthX = static_cast<int>(depthSpacePoints[colorIndex].X + 0.5f);
					const int depthY = static_cast<int>(depthSpacePoints[colorIndex].Y + 0.5f);
					if ((0 <= depthX) && (depthX < dWidth) && (0 <= depthY) && (depthY < dHeight)) {
						const unsigned int depthIndex = depthY * dWidth + depthX;
						buffer[colorIndex] = depthBuffer[depthIndex];
					}
				}
			});

			// Get Joint Data via Body Data
			cv::Mat colorJoints = color.clone();
			if (SUCCEEDED(hResultBody)) {
				IBody* pBody[BODY_COUNT] = { 0 }; // BODY_COUNT is 6
				pBodyFrame->GetAndRefreshBodyData(_countof(pBody), pBody);
				for (unsigned int bodyIndex = 0; bodyIndex < BODY_COUNT; bodyIndex++) {
					IBody *body = pBody[bodyIndex];

					// Skip joints if not being tracked
					BOOLEAN isTracked = false;
					HRESULT hrTrack = body->get_IsTracked(&isTracked);
					if (FAILED(hrTrack) || isTracked == false) {
						continue;
					}

					Joint joints[JointType_Count];
					HRESULT hr = body->GetJoints(_countof(joints), joints);
					const int radius = 75;
					const cv::Vec3b blue = cv::Vec3b(128, 0, 0), green = cv::Vec3b(0, 128, 0), red = cv::Vec3b(0, 0, 128);

					// Draw hand joints if present
					if (SUCCEEDED(hr)) {
						std::cout << "Successful Get Joints" << std::endl;
						const CameraSpacePoint &leftHandPos = joints[JointType_HandLeft].Position;
						std::cout << leftHandPos.X << ", " << leftHandPos.Y << ", " << leftHandPos.Z << std::endl;
						cv::Point handLeft = drawEllipse(colorJoints, joints[JointType_HandLeft], radius, green, 5);
						cv::Point handTipLeft = drawEllipse(colorJoints, joints[JointType_HandTipLeft], radius, green, 5);
						cv::Point handRight = drawEllipse(colorJoints, joints[JointType_HandRight], radius, green, 5);
						cv::Point handTipRight = drawEllipse(colorJoints, joints[JointType_HandTipRight], radius, green, 5);

						if (toggleSave) {
							if (curFrameCount % SAVE_INT == 0) {
								file << curFrameCount << "," << bodyIndex << "," << handLeft.x     << "," << handLeft.y     << "," <<
																					handTipLeft.x  << "," << handTipLeft.y  << "," <<
																					handRight.x    << "," << handRight.y    << "," <<
																					handTipRight.x << "," << handTipRight.y << "," << std::endl;
							}
						}
					}
				}
				SafeRelease(pBodyFrame);
			}

			// Create cv::Mat from Coordinate 
			cv::Mat depthAligned = cv::Mat(cHeight, cWidth, CV_16UC1, &buffer[0]).clone();

			// Convert to readable color code; we're using a fixed depth range here
			double dmin;
			double dmax;
			dmin = min;
			dmax = 1000;
			cv::Mat colorDepthAligned;
			depthAligned.convertTo(colorDepthAligned, CV_8UC1, 255 / (dmax - dmin), -dmin);
			cv::applyColorMap(colorDepthAligned, colorDepthAligned, cv::COLORMAP_JET);

			cv::imshow("Joints", colorJoints);
			cv::imshow("Color", color);
			cv::imshow("Depth", depth);
			cv::imshow("Depth aligned", colorDepthAligned);

			// Add images to vector to prepare for saving
			if (toggleSave) {
				if (curFrameCount % SAVE_INT == 0) {
					estEnd = std::chrono::system_clock::now();
					std::cout << "Saved!" << std::endl;
					std::chrono::duration<double, std::milli> fp_ms = estEnd - estBegin;
					std::cout << fp_ms.count() << std::endl;
					tfile << curFrameCount << "," << fp_ms.count() << std::endl;

					colorCache.push_back(color.clone());
					depthCache.push_back(depthAligned);

					frameCount++;
					key_edge = false;
				}
			}
			curFrameCount++;
		}

		SafeRelease(pColorFrame);
		SafeRelease(pDepthFrame);
		SafeRelease(pBodyFrame);

		key = cv::waitKey(30);

		// This section handles key input
		if (key == 'd' || key == 'd') {
			key_edge = true;
		}
		if (key == VK_ESCAPE) {
			break;
		} 

		// Handles saving of data
		else if (key == 'S' || key == 's')
		{
			if (toggleSave == false) {
				toggleSave = true;
				frameCount = 0;
				curFrameCount = 0;

				std::string folder = PATH_NAME + DATASET_NAME + std::to_string(datasetCount);

				if (!std::experimental::filesystem::exists(folder)) {
					std::experimental::filesystem::remove_all(folder);
					std::experimental::filesystem::create_directories(folder);
					std::experimental::filesystem::create_directories(folder + "/depth");
					std::experimental::filesystem::create_directories(folder + "/rgb");
				}

				file.open(folder + "/joints.csv");
				tfile.open(folder + "/timestamps.csv");
				std::cout << "WRITING" << std::endl;
			}
			else {
				txtWriterColor.open(PATH_NAME + DATASET_NAME + std::to_string(datasetCount) + "/" + "color.txt");
				txtWriterDepth.open(PATH_NAME + DATASET_NAME + std::to_string(datasetCount) + "/" + "depth.txt");

				for (int i = 0; i < frameCount; i++) {
					txtWriterColor << COLOR_NAME + std::to_string(i) + ".png" << std::endl;
					txtWriterDepth << DEPTH_NAME + std::to_string(i) + ".png" << std::endl;

					cv::imwrite(PATH_NAME + DATASET_NAME + std::to_string(datasetCount) + "/rgb/" + COLOR_NAME + std::to_string(i) + ".png", colorCache[i], p);
					cv::imwrite(PATH_NAME + DATASET_NAME + std::to_string(datasetCount) + "/depth/" + DEPTH_NAME + std::to_string(i) + ".png", depthCache[i], p);
				}
				txtWriterColor.close();
				txtWriterDepth.close();
				toggleSave = false;
				datasetCount++;

				file.close();
				tfile.close();

				colorCache.clear();
				depthCache.clear();
				std::cout << "END WRITING" << std::endl;
			}
		}
	}
	return 0;
}