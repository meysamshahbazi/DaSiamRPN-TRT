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


using namespace cv;
using namespace std;

void __add_da();

#endif