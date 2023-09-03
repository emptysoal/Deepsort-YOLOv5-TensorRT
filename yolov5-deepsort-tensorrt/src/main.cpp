#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <cmath>
#include <time.h>
#include <opencv2/opencv.hpp>

#include "manager.hpp"

using namespace cv;


const char *      yoloEngine = "../resources/model.plan";
const char *      sortEngine = "../resources/deepsort.plan";
const std::string videoPath = "../test_videos/demo.mp4";
int               gpuId = 0;
// 需要跟踪的类别，可以根据自己需求调整，筛选自己想要跟踪的对象的种类（以下对应COCO数据集类别索引）
std::vector<int>  trackClasses {0, 1, 2, 3, 5, 7};  // person, bicycle, car, motorcycle, bus, truck


int main(){

	Trtyolosort yosort(yoloEngine, sortEngine, gpuId, trackClasses);

	VideoCapture capture;
	cv::Mat frame;
	frame = capture.open(videoPath);
	if (!capture.isOpened()){
		std::cout<<"can not open"<<std::endl;
		return -1 ;
	}

	int i = 0;
	while(capture.read(frame)){
		if (i % 3 == 0){
		//std::cout<<"origin img size:"<<frame.cols<<" "<<frame.rows<<std::endl;
		auto start = std::chrono::system_clock::now();
		yosort.TrtDetect(frame);
		auto end = std::chrono::system_clock::now();
		int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
		}
		i++;
	}
	capture.release();
	return 0;

}
