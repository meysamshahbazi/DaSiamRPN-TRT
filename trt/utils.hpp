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

void printDim(const nvinfer1::Dims& dims);

void get_crop_single(Mat & im,Point2f target_pos_,
                                float target_scale,int output_sz,Scalar avg_chans,
                                Mat &im_patch,float &real_scale); // these are output 


void get_subwindow_tracking(const Mat &im, Point pos,int model_sz,int original_sz,Scalar avg_chans,Mat &z_crop);
class Logger : public nvinfer1::ILogger
{
void log(Severity severity, const char* msg) noexcept override
{
    // suppress info-level messages
    // if (severity <= Severity::kWARNING)
    //     std::cout << msg << std::endl;
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

void postprocessResults(float * gpu_output,const nvinfer1::Dims &dims, int batch_size, std::string file_name);

std::vector<vector<float>> xyxy2cxywh(float *box);

// anchor = generate_anchor(total_stride=8, scales= [8, ], 
                        //  ratios = [0.33, 0.5, 1, 2, 3], score_size =int(score_size))


std::vector< vector<float> > generate_anchor(int total_stride, float scale, std::vector<float> ratios, int score_size);


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