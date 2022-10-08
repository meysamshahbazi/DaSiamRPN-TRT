
#ifndef __DASIAM_HPP__
#define __DASIAM_HPP__
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
#include <algorithm>
#include "utils.hpp"
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

class DaSiam
{
private:
    // most of const variable are tracker configuration and the must be intialized corectly
    const int exemplar_size = 127; // input z size
    const int instance_size = 271; // input x size (search region)
    const int total_stride = 8;
    const float scales = 8.0f;

    const int score_size = (instance_size-exemplar_size)/total_stride+1;
    const float context_amount = 0.5; // context amount for the exemplar
    const bool adaptive = true; 
    const float penalty_k = 0.22;
    const float window_influence = 0.40;
    const float p_lr = 0.3;
    const string  temple_path{"../../temple.onnx"};
    const string  siam_path{"../../SiamRPNOTB.onnx"};
    const string  regress_path{"../../RegressAdjust.onnx"};

    const string  temple_path_engine{"../../temple.engine"};
    const string  siam_path_engine{"../../SiamRPNOTB.engine"};
    const string  regress_path_engine{"../../RegressAdjust.engine"};

    Logger logger;

    // end of config -----------------------------------------------------------------
    Point2f target_pos;
    Size2f target_sz;
    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_temple{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_temple{nullptr};
    vector<void *> buffers_temple;
    vector<nvinfer1::Dims> input_dims_temple;
    vector<nvinfer1::Dims> output_dims_temple;

    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_siam{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_siam{nullptr};
    vector<void *> buffers_siam;
    vector<nvinfer1::Dims> input_dims_siam;
    vector<nvinfer1::Dims> output_dims_siam;

    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_r1{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_r1{nullptr};
    vector<void *> buffers_r1;
    vector<nvinfer1::Dims> input_dims_r1;
    vector<nvinfer1::Dims> output_dims_r1;

    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_cls{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_cls{nullptr};
    vector<void *> buffers_cls;
    vector<nvinfer1::Dims> input_dims_cls;
    vector<nvinfer1::Dims> output_dims_cls;

    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine_regress{nullptr};
    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context_regress{nullptr};
    vector<void *> buffers_regress;
    vector<nvinfer1::Dims> input_dims_regress;
    vector<nvinfer1::Dims> output_dims_regress;

    int im_h;
    int im_w;

    Mat window;
    Scalar avg_chans;// this has 4 value and the order is not the same as in python

    std::vector<float > ratios ={0.33, 0.5, 1, 2, 3};
    std::vector< vector<float> >anchor;
    
    std::vector<float> pscore;
    std::vector<float> penalty;

    void create_fconv_r(unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);
            
    void create_fconv_cls(unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context);

    float change(float r);
    float sz(float w,float h);



public:
    DaSiam();
    ~DaSiam();
    void init(const Mat &im, const Rect2f state);
    Rect2f update(const Mat &im);
};


#endif