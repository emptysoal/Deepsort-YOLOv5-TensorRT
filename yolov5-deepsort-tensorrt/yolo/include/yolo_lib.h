#ifndef YOLOV5_LIB
#define YOLOV5_LIB

#include "public.h"

using namespace nvinfer1;


struct DetectResult
{
    cv::Rect tlwh;  // top left width height
    float conf;
    int class_id;
};


class YoloDetecter
{
public:
    YoloDetecter(const std::string trtFile, int gpuId);
    ~YoloDetecter();
    std::vector<DetectResult> inference(cv::Mat& img);

private:
    ICudaEngine* getEngine();
    void inference();

private:
    YoloLogger *        gLogger;
    std::string         trtFile_;

    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    std::vector<int>    vTensorSize;  // bytes of input and output
    float *             inputData;
    float *             outputData;
    std::vector<void *> vBufferD;
};

#endif  // YOLOV5_LIB
