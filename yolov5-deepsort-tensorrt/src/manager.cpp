#include "manager.hpp"

using namespace cv;

static Logger gLogger;


Trtyolosort::Trtyolosort(const char* yolo_engine_path, const char* sort_engine_path, 
	int gpu_id, std::vector<int>& track_classes)
{
	gpu_id_ = gpu_id;
	track_classes_ = track_classes;

	detecter = new YoloDetecter(std::string(yolo_engine_path), gpu_id_);
	DS = new DeepSort(sort_engine_path, 128, k_feature_dim, gpu_id_, &gLogger);
}

Trtyolosort::~Trtyolosort(){
	delete detecter;
	delete DS;
}

void Trtyolosort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1);
        //std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
		//std::string lbl = cv::format("ID:%d_x:%f_y:%f",(int)box.trackID,(box.x1+box.x2)/2,(box.y1+box.y2)/2);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
    }
    cv::imshow("img", temp);
    cv::waitKey(1);
}

bool Trtyolosort::isTrackingClass(int class_id){
	for (auto& c : track_classes_){
		if (class_id == c) return true;
	}
	return false;
}

void Trtyolosort::TrtDetect(cv::Mat& frame){
	// yolo detect
	std::vector<DetectResult> res = detecter->inference(frame);
	// rebuild output format, and choose given class id
	std::vector<DetectBox> det;
	for (long unsigned int j = 0; j < res.size(); j++)
	{
		cv::Rect r = res[j].tlwh;
		float conf = (float)res[j].conf;
		int class_id = (int)res[j].class_id;

		if (isTrackingClass(class_id)){
			DetectBox dd(r.x, r.y, r.x + r.width, r.y + r.height, conf, class_id);
			det.push_back(dd);
		}
	}
	// run deepsort tracking
	DS->sort(frame, det);
	showDetection(frame, det);
}
