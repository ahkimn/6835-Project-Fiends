#include <opencv2/opencv.hpp>
#include <iomanip>
#include <iostream>
#include <thread>
#include <future>

#include "../header/kinectIntegration.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	KinectIntegration ki;
	ki.kinectLoop();
}