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
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>

#include <iostream>
#include <fstream>
#include <string>
#include "calibrator.hpp"
#include <opencv2/imgproc.hpp>

using namespace cv;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// #include <opencv2/core/cuda_stream_accessor.hpp>
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

void get_crop_single(Mat & im,Point2f target_pos_,
                                float target_scale,int output_sz,Scalar avg_chans,
                                Mat &im_patch,float &real_scale); // these are output 

class Logger : public nvinfer1::ILogger
{
void log(Severity severity, const char* msg) noexcept override
{
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}
};// logger;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims);

void parseOnnxModel(    const string &model_path,
                        size_t pool_size,
                        unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                        unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);

void parseOnnxModelINT8(    const string &model_path,
                            size_t pool_size,
                            calibration_model cal_type,
                            unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                            unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);

void parseEngineModel(  const string &engine_file_path,
                        unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                        unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);
                        
void saveEngineFile(const string &onnx_path,
                    const string &engine_path);

void serializeOnnx2engine(std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> &engine,const string &model_path);

Mat get_hann_win(Size sz);

void postprocessResults(float * gpu_output,const nvinfer1::Dims &dims, int batch_size, std::string file_name);

std::vector<vector<float>> xyxy2cxywh(float *box);
/*
__global__ void fill_m(float * fg_bg,int * xyxy)
{

    int x1 = xyxy[0];
    int y1 = xyxy[1];
    int x2 = xyxy[2];
    int y2 = xyxy[3];

    if( threadIdx.x> x1-1 && threadIdx.x <x2+1 && threadIdx.y> y1-1 && threadIdx.y <y2+1 )
        fg_bg[blockDim.x*threadIdx.y+threadIdx.x ] = 1.0;
    else 
        fg_bg[blockDim.x*threadIdx.y+threadIdx.x ] = 0.0;

    
    return;
}
*/

#endif