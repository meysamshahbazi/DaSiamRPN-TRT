#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudnn.h>


using namespace cv;
using namespace std;

struct  TRTDestroy
{
    template<class T>
    void operator()(T* obj) const 
    {
        if (obj)
            delete obj;
            // obj->destroy();
    }
};

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void __add_da();

void printDim(const nvinfer1::Dims& dims);

void get_subwindow_tracking(const Mat &im, Point pos,int model_sz,int original_sz,Scalar avg_chans,Mat &z_crop);
class Logger : public nvinfer1::ILogger
{
void log(Severity severity, const char* msg) noexcept override
{
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
    return;
}
};// logger;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims);

void parseOnnxModel(    const string &model_path,
                        size_t pool_size,
                        unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                        unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);

void parseEngineModel(  const string &engine_file_path,
                        unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                        unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);
                        
void saveEngineFile(const string &onnx_path,
                    const string &engine_path);

Mat get_hann_win(Size sz);

std::vector< vector<float> > generate_anchor(int total_stride, float scale, std::vector<float> ratios, int score_size);

void blobFromImage(cv::Mat& img, float* blob);

#endif