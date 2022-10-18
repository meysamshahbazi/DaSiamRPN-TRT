#ifndef __D_UTILS_H__
#define __D_UTILS_H__
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

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/async/transform.h>

using namespace cv;
using namespace std;

void __add_da();

/// @brief post ptocess
/// @param d_delta 
/// @param d_ancher 
/// @param d_score 
/// @param d_window 
/// @param window_influence 
/// @param w 
/// @param h 
/// @param penalty_k 
/// @param row_size 
/// @param ret host pointer with size 6 that contains res_x,res_y,res_w,res_h,penalty,score 
void foo( float* d_delta,float* d_ancher, float* d_score,float* d_window,
          float window_influence,float w,float h, float penalty_k,int row_size,
          float* ret,cudaStream_t stream);

#endif