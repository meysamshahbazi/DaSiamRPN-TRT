#include "utils.hpp"


Mat get_hann_win(Size sz)
{
    Mat hann_rows = Mat::ones(sz.height, 1, CV_32F);
    Mat hann_cols = Mat::ones(1, sz.width, CV_32F);
    int NN = sz.height - 1;
    if(NN != 0) {
        for (int i = 0; i < hann_rows.rows; ++i) {
            hann_rows.at<float>(i,0) = (float)(1.0/2.0 * (1.0 - cos(2*CV_PI*i/NN)));
        }
    }
    NN = sz.width - 1;
    if(NN != 0) {
        for (int i = 0; i < hann_cols.cols; ++i) {
            hann_cols.at<float>(0,i) = (float)(1.0/2.0 * (1.0 - cos(2*CV_PI*i/NN)));
        }
    }
    return hann_rows * hann_cols;
}

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

void printDim(const nvinfer1::Dims& dims)
{
    cout<<"[";
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        cout<<dims.d[i]<<", ";
    }
    cout<<" ]\n";
}

void get_subwindow_tracking(const Mat &img, Point pos,int model_sz,int original_sz,Scalar avg_chans,Mat &im_patch)
{
    // this is opencv function ...
    Mat  dst;
    Size imgSize = img.size();
    float c = (original_sz + 1) / 2;
    float xMin = (float)cvRound(pos.x - c);
    float xMax = xMin + original_sz - 1;
    float yMin = (float)cvRound(pos.y - c);
    float yMax = yMin + original_sz - 1;

    int leftPad = (int)(fmax(0., -xMin));
    int topPad = (int)(fmax(0., -yMin));
    int rightPad = (int)(fmax(0., xMax - imgSize.width + 1));
    int bottomPad = (int)(fmax(0., yMax - imgSize.height + 1));

    xMin = xMin + leftPad;
    xMax = xMax + leftPad;
    yMax = yMax + topPad;
    yMin = yMin + topPad;

    if (topPad == 0 && bottomPad == 0 && leftPad == 0 && rightPad == 0)
    {
        img(Rect(int(xMin), int(yMin), int(xMax - xMin + 1), int(yMax - yMin + 1))).copyTo(im_patch);
    }
    else
    {
        copyMakeBorder(img, dst, topPad, bottomPad, leftPad, rightPad, BORDER_CONSTANT, avg_chans);
        dst(Rect(int(xMin), int(yMin), int(xMax - xMin + 1), int(yMax - yMin + 1))).copyTo(im_patch);
    }
    cv::resize(im_patch,im_patch,cv::Size(model_sz,model_sz));
}
void parseOnnxModel(const string & onnx_path,
                    size_t pool_size,
                    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context)
{
    Logger logger;
    // first we create builder 
    unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
    // then define flag that is needed for creating network definitiopn 
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
    // then parse network 
    unique_ptr<nvonnxparser::IParser,TRTDestroy> parser{nvonnxparser::createParser(*network,logger)};
    // parse from file
    parser->parseFromFile(onnx_path.c_str(),static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    // lets create config file for engine 
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,pool_size);
    // config->setMaxWorkspaceSize(1U<<30);

    // use fp16 if it is possible 
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );
    context.reset(engine->createExecutionContext());
    return;
}

void saveEngineFile(const string & onnx_path,
                    const string & engine_path)
{
    Logger logger;
    // first we create builder 
    unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
    // then define flag that is needed for creating network definitiopn 
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
    // then parse network 
    unique_ptr<nvonnxparser::IParser,TRTDestroy> parser{nvonnxparser::createParser(*network,logger)};
    // parse from file
    parser->parseFromFile(onnx_path.c_str(),static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    // lets create config file for engine 
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1U<<30);
    // config->setMaxWorkspaceSize(1U<<30);

    // use fp16 if it is possible 
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // create engine and excution context
    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    std::ofstream p(engine_path, std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    return;
}

void parseEngineModel(const string & engine_file_path,
                    unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context)
{
    Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) 
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    assert(runtime != nullptr);
    // ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    engine.reset(runtime->deserializeCudaEngine(trtModelStream, size));
    assert(engine != nullptr); 
    context.reset(engine->createExecutionContext());
    assert(context != nullptr);
    delete[] trtModelStream;
    return;
}     

std::vector< vector<float> > generate_anchor(int total_stride, float scale, std::vector<float> ratios, int score_size)
{
    int anchor_num = ratios.size(); // we assume len(scale) = 1
    std::vector< vector<float> > anchor;
    int size = total_stride*total_stride;
    int count = 0;
    for(auto ratio:ratios)
    {
        for (int i=0;i<score_size*score_size;i++)
        {
            int ws = static_cast<int>( std::sqrt(size/ratio));
            int hs = static_cast<int>( ws*ratio );
            // becuse we have just one scale we skip FOR scale...
            // TODO: change this for vector of scale ...
            ws = ws*scale;
            hs = hs*scale;

            std::vector<float> elm;
            elm.push_back(0.0f);
            elm.push_back(0.0f);
            elm.push_back(ws);
            elm.push_back(hs);
            anchor.push_back(elm);
        }
    }

    int ori = -score_size *total_stride/ 2;
    int index = 0;
    for(int i=0; i<anchor_num; i++)
    {   
        for(int yy=0; yy<score_size; yy++)
        {
            for(int xx=0; xx<score_size; xx++)
            {
                anchor.at(index).at(0) = static_cast<float>(ori+total_stride*xx);
                anchor.at(index).at(1) = static_cast<float>(ori+total_stride*yy);
                index++;
            }
        }
    }

    return anchor;
}

void blobFromImage(cv::Mat& img, float* blob)
{
    int img_h = img.rows;
    int img_w = img.cols;
    int data_idx = 0;
    for (int i = 0; i < img_h; ++i)
    {
        uchar* pixel = img.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < img_w; ++j)
        {
            blob[data_idx+0*img_h*img_w] = static_cast<float>(*pixel++);
            blob[data_idx+1*img_h*img_w] = static_cast<float>(*pixel++);
            blob[data_idx+2*img_h*img_w] = static_cast<float>(*pixel++);
            data_idx++;
        }
    }
}
