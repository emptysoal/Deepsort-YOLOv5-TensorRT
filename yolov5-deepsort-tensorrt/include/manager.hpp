#ifndef _MANAGER_H
#define _MANAGER_H

#include <iostream>
#include <fstream>
#include <vector>
#include "logging.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "time.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "yolo_lib.h"
#include "deepsort.h"

using std::vector;
using namespace cv;
//static Logger gLogger;

class Trtyolosort{
public:
	// init 
	Trtyolosort(const char* yolo_engine_path, const char* sort_engine_path, int gpu_id, std::vector<int>& track_classes);
	~Trtyolosort();
	// detect and show
	void TrtDetect(cv::Mat &frame);
	void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes);

private:
	YoloDetecter *   detecter;
    DeepSort *       DS;

	int              gpu_id_;
	std::vector<int> track_classes_;

private:
	bool isTrackingClass(int class_id);
};

#endif  // _MANAGER_H

