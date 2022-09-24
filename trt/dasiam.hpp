
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

    const int score_size = (instance_size-exemplar_size)/total_stride+1;
    const float context_amount = 0.5; // context amount for the exemplar
    const bool adaptive = true; 
    const float penalty_k = 0.055;
    const float window_influence = 0.42;
    const float lr = 0.295;
    // end of config -----------------------------------------------------------------
    Point2f target_pos;
    Size2f target_sz;
    
    int im_h;
    int im_w;

    Mat window;
    Scalar avg_chans;// this has 4 value and the order is not the same as in python


public:
    DaSiam();
    ~DaSiam();
    void init(const Mat &im, const Rect2f state);
    Rect2f update(const Mat &im);
};


#endif