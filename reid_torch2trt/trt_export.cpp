#include "public.h"
#include <NvOnnxParser.h>

using namespace nvinfer1;


const int         inputHeight = 128;
const int         inputWidth = 64;
const int         maxBatchSize = 128;
const std::string onnxFile = "./deepsort.onnx";
const std::string trtFile = "./deepsort.plan";

static Logger     gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool        bFP16Mode = true;


int getEngine()
{
    IBuilder *            builder     = createInferBuilder(gLogger);
    INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile* profile     = builder->createOptimizationProfile();
    IBuilderConfig *      config      = builder->createBuilderConfig();

    if (bFP16Mode)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    config->setMaxWorkspaceSize(1 << 20);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity)))
    {
        std::cout << std::string("Failed parsing .onnx file!") << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            auto *error = parser->getError(i);
            std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
        }
        return 1;
    }
    std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

    ITensor* inputTensor = network->getInput(0);
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {maxBatchSize, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {maxBatchSize, 3, inputHeight, inputWidth}});
    config->addOptimizationProfile(profile);

    IHostMemory* engineString = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Succeeded building serialized engine!" << std::endl;

    std::ofstream engineFile(trtFile, std::ios::binary);
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    std::cout << "Succeeded saving .plan file!" << std::endl;

    return 0;
}


int main()
{
    CHECK(cudaSetDevice(0));
    if ( getEngine() != 0 ) return 1 ;

    std::cout << "==============" << std::endl;
    std::cout << "|  SUCCESS!  |" << std::endl;
    std::cout << "==============" << std::endl;

    return 0;
}
