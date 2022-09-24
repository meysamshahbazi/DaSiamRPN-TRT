
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

public:
    DaSiam();
    ~DaSiam();
    void init(const Mat &im, const Rect2f state);
    Rect2f update(const Mat &im);
};


#endif