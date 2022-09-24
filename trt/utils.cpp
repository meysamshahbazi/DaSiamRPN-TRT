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

void get_crop_single(Mat & im,Point2f target_pos_,
                                float target_scale,int output_sz,Scalar avg_chans,
                                Mat &im_patch,float &real_scale) // these are output 
{
    // reversed pos!! 
    // pos is target_pos
    Point2i posl = Point2i(target_pos_);
    float sample_sz = target_scale*output_sz;    
    float resize_factor = sample_sz/output_sz;
    
    int df = std::max(static_cast<int>(resize_factor-0.1),1);
    Mat im2;

    if (df > 1)
    {
        // Point2i os = Point2i(static_cast<int>(target_pos_.x) % df,static_cast<int>(target_pos_.y) % df);
        Point2i os = Point2i( posl.x % df, posl.y % df);
        std::vector<uchar> im_rows;
        std::vector< std::vector<uchar> > im_cols;
        posl = Point2i((posl.x-os.x)/df , (posl.y-os.y)/df);
        int channels = im.channels();

        int nRows = im.rows;
        int nCols = im.cols * channels;
        int i,j;
        uchar* p;
        std::vector<uchar> pixel_vec;
        
        int reduced_row_sz = (im.rows - os.x+df-1)/df;
        int reduced_col_sz = (im.cols - os.y+df-1)/df;

        uchar img_data[reduced_row_sz*reduced_col_sz*3];
        int data_idx = 0;
        for( i = os.x; i < nRows; i+=df)
        {
            uchar* pixel = im.ptr<uchar>(i);  // point to first color in row
            for ( j = os.y*3; j < nCols;j+=3*df)
            {
                img_data[data_idx] = *pixel++;
                img_data[data_idx+1] = *pixel++;
                img_data[data_idx+2] = *pixel++;
                data_idx+=3;
                pixel += 3*df-3;
            }
        }
        im2 = cv::Mat(reduced_row_sz, reduced_col_sz, CV_8UC3,img_data);
    }
    else
    {
        im2 = im;
    }

    float sz = sample_sz/df;
    
    int szl = std::max(static_cast<int>(std::round(sz)) ,2);

    Point2i tl = Point2i(posl.x-(szl-1)/2,posl.y-(szl-1)/2);
    Point2i br = Point2i(posl.x+szl/2+1,posl.y+szl/2+1);

    float M_13 = tl.x;
    float M_23 = tl.y;
    float M_11 = (br.x-M_13)/(output_sz-1);
    float M_22 = (br.y-M_23)/(output_sz-1);

    float arr[2][3] = { 
                        {M_11,0,M_13},
                        {0,M_22,M_23}
                        };

    Mat mat2x3 = Mat(2,3,CV_32FC1 ,arr);

    cv::warpAffine( im2,
                    im_patch,
                    mat2x3,
                    Size(output_sz,output_sz),
                    INTER_LINEAR | WARP_INVERSE_MAP,
                    BORDER_CONSTANT,
                    avg_chans
    ); 	

    real_scale = static_cast<float>(output_sz)/((br.x-tl.x+1)*df) ;   
}


void parseOnnxModelINT8(const string & onnx_path,
                    size_t pool_size,
                    calibration_model cal_type,
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
    // lets create config file for engine 
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,pool_size);
    // config->setMaxWorkspaceSize(1U<<30);
    std::unique_ptr<nvinfer1::IInt8Calibrator,TRTDestroy> calibrator;
    if(cal_type == calibration_model::BASE_Q)
    {
        calibrator.reset(
        new MyCalibrator(2,
                    "/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/"
                    ,"calb_q.txt",cal_type) ) ;
    }
    
    if(cal_type == calibration_model::MEMORIZE)
    {
        calibrator.reset(
        new MyCalibrator(2,
                    "/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/"
                    ,"calb_m.txt",cal_type) ) ;
    }

    if (builder->platformHasFastInt8() )
    {
        std::cout<<"platformHasFastInt8 platformHasFastInt8 platformHasFastInt8 platformHasFastInt8"<<endl;
        // config->setFlsag(nvinfer1::BuilderFlag::kFP16);
    }
    // void * temp;
    // const char * nme = "img";
    // calibrator->getBatch(&temp,&nme,1);     

    config->setFlag(nvinfer1::BuilderFlag::kINT8);           
    config->setInt8Calibrator(calibrator.get());
    
    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    cout<<"im here"<<endl;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);

    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );

    // serializedModel->
    // auto tmp = builder->buildEngineWithConfig(*network,*config);
    // engine.reset(builder->buildEngineWithConfig(*network,*config));
    // engine.reset(builder->buildSerializedNetwork(*network,*config));
    context.reset(engine->createExecutionContext());
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
    

    // builder->platformHasFastInt8
    
    // config->setInt8Calibrator

    // setm max bach size as it is very importannt for trt
    // builder->setMaxBatchSize(1);
    // create engine and excution context
    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);


    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );



    // serializedModel->
    // auto tmp = builder->buildEngineWithConfig(*network,*config);
    // engine.reset(builder->buildEngineWithConfig(*network,*config));
    // engine.reset(builder->buildSerializedNetwork(*network,*config));
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

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1U<<24);
    // config->setMaxWorkspaceSize(1U<<30);

    // use fp16 if it is possible 

    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // setm max bach size as it is very importannt for trt
    // builder->setMaxBatchSize(1);
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



void postprocessResults(float * gpu_output,const nvinfer1::Dims &dims, int batch_size, std::string file_name)
{
    // auto classes = getClassNames("../imagenet_classes.txt");

    vector<float> cpu_output(getSizeByDim(dims)*batch_size);
    cudaMemcpy(cpu_output.data(),gpu_output,cpu_output.size()*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<"size : "<<cpu_output.size()<<endl;

    std::ofstream out_file{file_name + ".txt"};
    for (const auto &v :cpu_output)
            out_file << v << std::endl;

    out_file.close();


}

// std::vector<vector<float>> xyxy2cxywh(const std::vector<float> & box)
std::vector<vector<float>> xyxy2cxywh(float * box)
{
    std::vector<vector<float>> box_wh;
    for(int i = 0;i < 625; i++)
    {
        float cx = (box[4*i+0]+box[4*i+2])/2;
        float cy = (box[4*i+1]+box[4*i+3])/2;
        float w = box[4*i+2]-box[4*i+0]+1;
        float h = box[4*i+3]-box[4*i+1]+1;
        box_wh.push_back({cx,cy,w,h});
    } 
    return box_wh;
}


