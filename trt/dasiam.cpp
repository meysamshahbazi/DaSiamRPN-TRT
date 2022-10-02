#include "dasiam.hpp"

// #define GEN_ENGINE_FROM_ONNX
#define LOAD_FROM_ONNX
// #define LOAD_FROM_ENGINE

DaSiam::DaSiam()
{
#ifdef GEN_ENGINE_FROM_ONNX
    saveEngineFile(temple_path,temple_path_engine);
    saveEngineFile(siam_path,siam_path_engine);
    cout<<"finished serialization\n";
    return;
#endif
#ifdef LOAD_FROM_ONNX
    parseOnnxModel(temple_path,1U<<24,engine_temple,context_temple);
    parseOnnxModel(siam_path,1U<<24,engine_siam,context_siam);
    parseOnnxModel(regress_path,1U<<24,engine_regress,context_regress);
#endif
#ifdef LOAD_FROM_ENGINE
    parseEngineModel(temple_path_engine,engine_temple,context_temple);
    parseEngineModel(siam_path,engine_siam,context_siam);
#endif    

    buffers_temple.reserve(engine_temple->getNbBindings());
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_temple->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_temple->getBindingDimensions(i)) * 1 * sizeof(float);
        cudaMalloc(&buffers_temple[i], binding_size);
        std::cout<<engine_temple->getBindingName(i)<<std::endl;
        printDim(engine_temple->getBindingDimensions(i));
        if (engine_temple->bindingIsInput(i))
        {  
            input_dims_temple.emplace_back(engine_temple->getBindingDimensions(i));
        }
        else
        {
            output_dims_temple.emplace_back(engine_temple->getBindingDimensions(i));
        }
    }

    
    buffers_siam.reserve(engine_siam->getNbBindings());
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_siam->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_siam->getBindingDimensions(i)) * 1 * sizeof(float);
        cudaMalloc(&buffers_siam[i], binding_size);
        std::cout<<engine_siam->getBindingName(i);//"|" <<engine_siam->getBindingDimensions(i)<<std::endl;
        printDim(engine_siam->getBindingDimensions(i));
        if (engine_siam->bindingIsInput(i))
        {  
            input_dims_siam.emplace_back(engine_siam->getBindingDimensions(i));
        }
        else
        {
            output_dims_siam.emplace_back(engine_siam->getBindingDimensions(i));
        }
    }

    buffers_regress.reserve(engine_regress->getNbBindings());
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_regress->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_regress->getBindingDimensions(i)) * 1 * sizeof(float);
        cudaMalloc(&buffers_regress[i], binding_size);
        std::cout<<engine_regress->getBindingName(i);//"|" <<engine_regress->getBindingDimensions(i)<<std::endl;
        printDim(engine_regress->getBindingDimensions(i));
        if (engine_regress->bindingIsInput(i))
        {  
            input_dims_regress.emplace_back(engine_regress->getBindingDimensions(i));
        }
        else
        {
            output_dims_regress.emplace_back(engine_regress->getBindingDimensions(i));
        }
    }



    size_t binding_size;
    buffers_r1.reserve(2);
    binding_size = 256*22*22*sizeof(float);
    cudaMalloc(&buffers_r1[0], binding_size);
    binding_size = 20*19*19*sizeof(float);
    cudaMalloc(&buffers_r1[1], binding_size);

    buffers_cls.reserve(2);
    binding_size = 256*22*22*sizeof(float);
    cudaMalloc(&buffers_cls[0], binding_size);
    binding_size = 10*19*19*sizeof(float);
    cudaMalloc(&buffers_cls[1], 2);
}

DaSiam::~DaSiam()
{
    for (void * buf : buffers_temple)
        cudaFree(buf);
    for (void * buf : buffers_siam)
        cudaFree(buf);

    for (void * buf : buffers_r1)
        cudaFree(buf);
    
    for (void * buf : buffers_cls)
        cudaFree(buf);


}

void blobFromImage(cv::Mat& img,float* blob){
    int img_h = img.rows;
    int img_w = img.cols;
    int data_idx = 0;
    for (int i = 0; i < img_h; ++i)
    {
        uchar* pixel = img.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < img_w; ++j)
        {
            blob[data_idx] = (*pixel++);
            blob[data_idx+img_h*img_w] = (*pixel++);
            blob[data_idx+2*img_h*img_w] = (*pixel++);
            data_idx++;
        }
    }
}


void DaSiam::init(const Mat &im, const Rect2f state)
{
    target_pos = (state.br()+state.tl())/2;
    target_sz = state.size();
    im_h = im.rows;
    im_w = im.cols;
    // there is some stuf in init fucntion in python for small object!!!!
    // in this stage I've ignored that :D

    // p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))
    window = get_hann_win(Size(score_size,score_size));
    avg_chans = cv::mean(im);

    float wc_z = target_sz.width + context_amount*(target_sz.width+target_sz.height);
    float hc_z = target_sz.height + context_amount*(target_sz.width+target_sz.height);

    int s_z = std::round<int>(std::sqrt(wc_z*hc_z));

    Mat z_crop;
    get_subwindow_tracking(im,target_pos,exemplar_size,s_z,avg_chans,z_crop);
    float * temple_blob = new float[3*exemplar_size*exemplar_size];
    blobFromImage(z_crop,temple_blob);
    cudaMemcpyAsync(buffers_temple[0], temple_blob, 3 * exemplar_size * exemplar_size * sizeof(float), cudaMemcpyHostToDevice);
    context_temple->enqueueV2(buffers_temple.data(), 0, nullptr);
    cudaStreamSynchronize(0);
    create_fconv_r(engine_r1,context_r1);
    create_fconv_cls(engine_cls,context_cls);



    // cout<<z_crop.size()<<endl;
    // cv::imshow("z_crop",z_crop);
    
    return;
}

Rect2f DaSiam::update(const Mat &im)
{
    float wc_z = target_sz.width + context_amount*(target_sz.width+target_sz.height);
    float hc_z = target_sz.height + context_amount*(target_sz.width+target_sz.height);
    float s_z = std::sqrt(wc_z*hc_z);
    float scale_z = exemplar_size/s_z; 
    float d_search = (instance_size-exemplar_size)/2;
    float pad = d_search/scale_z;
    int s_x = std::round<int>(s_z+2*pad);
    Mat x_crop;
    get_subwindow_tracking(im,target_pos,instance_size,s_x,avg_chans,x_crop);
    // def tracker_eval function
    float* blob = new float[3*instance_size*instance_size];
    blobFromImage(x_crop,blob);

    // cv::imshow("z_crop",x_crop);
    // cv::waitKey(0);
    cudaMemcpyAsync(buffers_siam[0], blob, 3 * instance_size * instance_size * sizeof(float), cudaMemcpyHostToDevice);
    context_siam->enqueueV2(buffers_siam.data(), 0, nullptr);
    buffers_r1[0] = buffers_siam[1]; // delta
    buffers_cls[0] = buffers_siam[2]; // score
    context_r1->enqueueV2(buffers_r1.data(), 0, nullptr);
    context_cls->enqueueV2(buffers_cls.data(), 0, nullptr); // TODO do it with cuDNN
    buffers_regress[0] = buffers_r1[1];
    context_regress->enqueueV2(buffers_regress.data(), 0, nullptr);
    // these are for F.conv...  
    


    // cv::waitKey(0);
    return Rect2f(0,0,100,100);// dummy return just for test
}

void DaSiam::create_fconv_r(unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context)
{
    unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
    unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
    nvinfer1::ITensor* data = network->addInput("delta", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1,256, 22, 22});
    // unique_ptr<nvinfer1::ITensor,TRTDestroy> data{network->addInput("delta", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{256, 22, 22})};
    size_t kernel_size = 20*256*4*4;
    float *kernel_r_wt = new float[kernel_size];
    // cudaMalloc(&kernel_r_wt,20*256*4*4*sizeof(float));
    cudaMemcpy(kernel_r_wt,buffers_temple[1],kernel_size*sizeof(float),cudaMemcpyDeviceToHost);
    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT,(void*)kernel_r_wt, kernel_size};

    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 20, nvinfer1::DimsHW{4, 4}, wt, nvinfer1::Weights{});
    
    // unique_ptr<nvinfer1::IConvolutionLayer,TRTDestroy> conv1{network->addConvolutionNd(*data, 20, nvinfer1::DimsHW{4, 4}, wt, nvinfer1::Weights{})};
    conv1->setStrideNd(nvinfer1::DimsHW{1, 1});
    
    conv1->getOutput(0)->setName("fconv_r1_kernel");
    
    network->markOutput(*conv1->getOutput(0));
    
    // builder->setMaxBatchSize(1);
    // config->setMaxWorkspaceSize();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1 << 30);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    
    // nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // engine.reset(builder->buildEngineWithConfig(*network, *config));
    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    // cout<<"conv1: "<<conv1<<endl;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );

    context.reset(engine->createExecutionContext());
}

void DaSiam::create_fconv_cls(unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context)
{
    unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
    nvinfer1::ITensor* data = network->addInput("score", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1,256, 22, 22});

    size_t kernel_size = 10*256*4*4;
    float *kernel_cls_wt = new float[kernel_size];
    // cudaMalloc(&kernel_r_wt,20*256*4*4*sizeof(float));
    cudaMemcpy(kernel_cls_wt,buffers_temple[2],kernel_size*sizeof(float),cudaMemcpyDeviceToHost);
    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT,(void*)kernel_cls_wt, kernel_size};

    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 10, nvinfer1::DimsHW{4, 4}, wt, nvinfer1::Weights{});
    
    conv1->setStrideNd(nvinfer1::DimsHW{1, 1});
    
    conv1->getOutput(0)->setName("fconv_cls_kernel");
    
    network->markOutput(*conv1->getOutput(0));

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1 << 30);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );

    context.reset(engine->createExecutionContext());
}