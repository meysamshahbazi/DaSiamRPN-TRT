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
    cudaMalloc(&buffers_cls[1], binding_size);

    anchor = generate_anchor(total_stride,scales,ratios,score_size);
    window = get_hann_win(Size(score_size,score_size));
    delta = new float[4*anchor.size()];
    score = new float[2*anchor.size()];
    blob = new float[3*instance_size*instance_size];
    temple_blob = new float[3*exemplar_size*exemplar_size];
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
    for (void * buf : buffers_regress)
        cudaFree(buf);

    delete[] score;
    delete[] delta;
    delete[] blob;
    delete[] temple_blob;
}

void DaSiam::init(const Mat &im, const Rect2f state)
{
    target_pos = (state.br()+state.tl())/2;
    target_sz = state.size();
    im_h = im.rows;
    im_w = im.cols;
    avg_chans = cv::mean(im);
    float wc_z = target_sz.width + context_amount*(target_sz.width+target_sz.height);
    float hc_z = target_sz.height + context_amount*(target_sz.width+target_sz.height);
    int s_z = std::round<int>(std::sqrt(wc_z*hc_z));
    Mat z_crop;
    get_subwindow_tracking(im,target_pos,exemplar_size,s_z,avg_chans,z_crop);
    blobFromImage(z_crop,temple_blob);
    cudaMemcpyAsync(buffers_temple[0], temple_blob, 3 * exemplar_size * exemplar_size * sizeof(float), cudaMemcpyHostToDevice);
    context_temple->enqueueV2(buffers_temple.data(), 0, nullptr);
    cudaStreamSynchronize(0); // this is vital to wait for last results ..
    create_fconv_r(engine_r1,context_r1);
    create_fconv_cls(engine_cls,context_cls);

    // cout<<"------------------------------"<<endl;
    // for (size_t i = 0; i < engine_r1->getNbBindings(); ++i)
    // {
    //     auto binding_size = getSizeByDim(engine_r1->getBindingDimensions(i)) * 1 * sizeof(float);
    //     // cudaMalloc(&buffers_r1[i], binding_size);
    //     std::cout<<engine_r1->getBindingName(i)<<endl;//"|" <<engine_r1->getBindingDimensions(i)<<std::endl;
    //     printDim(engine_r1->getBindingDimensions(i));
    //     if (engine_r1->bindingIsInput(i))
    //     {  
    //         input_dims_r1.emplace_back(engine_r1->getBindingDimensions(i));
    //     }
    //     else
    //     {
    //         output_dims_r1.emplace_back(engine_r1->getBindingDimensions(i));
    //     }
    // }

    // // buffers_cls.reserve(engine_cls->getNbBindings());
    // cout<<"------------------------------"<<endl;
    // for (size_t i = 0; i < engine_cls->getNbBindings(); ++i)
    // {
    //     auto binding_size = getSizeByDim(engine_cls->getBindingDimensions(i)) * 1 * sizeof(float);
    //     // cudaMalloc(&buffers_cls[i], binding_size);
    //     std::cout<<engine_cls->getBindingName(i)<<endl;;//"|" <<engine_cls->getBindingDimensions(i)<<std::endl;
    //     printDim(engine_cls->getBindingDimensions(i));
    //     if (engine_cls->bindingIsInput(i))
    //     {  
    //         input_dims_cls.emplace_back(engine_cls->getBindingDimensions(i));
    //     }
    //     else
    //     {
    //         output_dims_cls.emplace_back(engine_cls->getBindingDimensions(i));
    //     }
    // }
    // cout<<"------------------------------"<<endl;
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
    int64 t1 = cv::getTickCount();
    get_subwindow_tracking(im,target_pos,instance_size,s_x,avg_chans,x_crop);
    // cv::imshow("x_crop",x_crop);cv::waitKey(0);
    //from this line: def tracker_eval function

    target_sz = target_sz*scale_z;
    blobFromImage(x_crop,blob);
    // for(int i = 0;i<3*instance_size*instance_size;i++) blob[i] = 255.0f; // fill with ones
    cudaMemcpyAsync(buffers_siam[0], blob, 3 * instance_size * instance_size * sizeof(float), cudaMemcpyHostToDevice);
    context_siam->enqueueV2(buffers_siam.data(), 0, nullptr);
    // cudaStreamSynchronize(0);
    buffers_r1[0]  = buffers_siam[1]; // delta
    buffers_cls[0] = buffers_siam[2]; // score
    // cudaMemcpyAsync(buffers_r1[0], buffers_siam[1], 256*22*22*sizeof(float), cudaMemcpyDeviceToDevice);
    // cudaMemcpyAsync(buffers_cls[0], buffers_siam[2], 256*22*22*sizeof(float), cudaMemcpyDeviceToDevice);
    context_r1->enqueueV2(buffers_r1.data(), 0, nullptr);
    context_cls->enqueueV2(buffers_cls.data(), 0, nullptr); // TODO do it with cuDNN
    buffers_regress[0] = buffers_r1[1];
    // cudaStreamSynchronize(0);
    // cudaMemcpyAsync(buffers_regress[0], buffers_r1[1], 20*19*19*sizeof(float), cudaMemcpyDeviceToDevice);
    context_regress->enqueueV2(buffers_regress.data(), 0, nullptr);
    int delta_size = anchor.size()*4;

    cudaMemcpyAsync(delta, buffers_regress[1], delta_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(score, buffers_cls[1], 2*anchor.size()*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaStreamSynchronize(0);

    // int64 t2 = cv::getTickCount();
    // int64 tick_counter = t2 - t1;
    // cout << "FPS: " << ((double)(1.0f)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;
    // delta in shape of [4,1805]
    
    pscore.clear(); // TODO:use fixed size array instead of vector
    penalty.clear();
    std::vector<float> temp_score;

    for(int i =0;i<anchor.size();i++)
    {
        std::vector<float> row = anchor.at(i);
        delta[i + 0*anchor.size()] = delta[i + 0*anchor.size()] * static_cast<float>(row.at(2)) + static_cast<float>(row.at(0)); 
        delta[i + 1*anchor.size()] = delta[i + 1*anchor.size()] * static_cast<float>(row.at(3)) + static_cast<float>(row.at(1)); 
        delta[i + 2*anchor.size()] = std::exp(delta[i + 2*anchor.size()]) * static_cast<float>(row.at(2));
        delta[i + 3*anchor.size()] = std::exp(delta[i + 3*anchor.size()]) * static_cast<float>(row.at(3));
        // softmax!:
        float score_ = std::exp(score[i+1*anchor.size()])/( std::exp(score[i+0*anchor.size()])+std::exp(score[i+1*anchor.size()]) );

        float s_c = change(
                    sz(delta[i + 2*anchor.size()],delta[i + 3*anchor.size()])/
                    sz(target_sz.width,target_sz.height)
        );

        float r_c = change( (target_sz.width/target_sz.height) /
                            (delta[i + 2*anchor.size()]/delta[i + 3*anchor.size()]));

        float penalty_ = std::exp(-(r_c * s_c - 1.0f) * penalty_k);
        float pscore_ = penalty_ * score_;
        int index_window = i%(score_size*score_size);
        pscore_ = pscore_*(1-window_influence)+ window.at<float>(index_window/score_size,index_window%score_size)*window_influence;
        pscore.push_back(pscore_);
        penalty.push_back(penalty_);
        temp_score.push_back(score_);
    }

    auto max_pscore_it = std::max_element(pscore.begin(),pscore.end());
    int best_pscore_id = distance(pscore.begin(), max_pscore_it);
    
    float res_x = delta[best_pscore_id + 0*anchor.size()]/scale_z;
    float res_y = delta[best_pscore_id + 1*anchor.size()]/scale_z;
    float res_w = delta[best_pscore_id + 2*anchor.size()]/scale_z;
    float res_h = delta[best_pscore_id + 3*anchor.size()]/scale_z;

    float lr = penalty[best_pscore_id]*temp_score[best_pscore_id]*p_lr;
  
    res_x = res_x + target_pos.x;
    res_y = res_y + target_pos.y;
    target_sz = target_sz / scale_z;
    res_w = target_sz.width*(1-lr)+res_w*lr;
    res_h = target_sz.height*(1-lr)+res_h*lr;
    // // TODO: store pscore
    res_x = std::max(0.0f, std::min( (float)im_w, res_x));
    res_y = std::max(0.0f, std::min( (float)im_h, res_y));
    res_w = std::max(min_w, std::min( (float)im_w, res_w));
    res_h = std::max(min_h, std::min( (float)im_h, res_h));

    target_pos = Point2f(res_x,res_y);
    target_sz = Size2f(res_w,res_h);
    // end of def tracker_eval function

    float x1 = target_pos.x-target_sz.width/2;
    float y1 = target_pos.y-target_sz.height/2;  
    Rect2f track_rect = Rect2f(x1, y1, target_sz.width, target_sz.height);
    return track_rect;
}

float DaSiam::change(float r)
{
    return std::max(r,1.0f/r);
}

float DaSiam::sz(float w,float h)
{
    float pad = (w+h)/2;
    float sz2 = (w+pad)*(h+pad);
    return std::sqrt(sz2);
}

void DaSiam::create_fconv_r(unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context)
{
    unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
    nvinfer1::ITensor* data = network->addInput("delta", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1,256, 22, 22});
    int64_t kernel_size = 20*256*4*4;
    float *kernel_r_wt = new float[kernel_size];
    cudaMemcpy(kernel_r_wt,buffers_temple[1],kernel_size*sizeof(float),cudaMemcpyDeviceToHost);
    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT,(void*)kernel_r_wt, kernel_size};
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 20, nvinfer1::DimsHW{4, 4}, wt, nvinfer1::Weights{});
    conv1->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv1->getOutput(0)->setName("fconv_r1_kernel");
    network->markOutput(*conv1->getOutput(0));
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1 << 24);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );
    context.reset(engine->createExecutionContext());
    delete[] kernel_r_wt;
}

void DaSiam::create_fconv_cls(unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> &engine,
                    unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> &context)
{
    unique_ptr<nvinfer1::IBuilder,TRTDestroy> builder{nvinfer1::createInferBuilder(logger)};
    unique_ptr<nvinfer1::IBuilderConfig,TRTDestroy> config{builder->createBuilderConfig()};
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<nvinfer1::INetworkDefinition,TRTDestroy> network{builder->createNetworkV2(flag)};
    nvinfer1::ITensor* data = network->addInput("score", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1,256, 22, 22});
    int64_t kernel_size = 10*256*4*4;
    float *kernel_cls_wt = new float[kernel_size];
    // cudaMalloc(&kernel_r_wt,20*256*4*4*sizeof(float));
    cudaMemcpy(kernel_cls_wt,buffers_temple[2],kernel_size*sizeof(float),cudaMemcpyDeviceToHost);
    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT,(void*)kernel_cls_wt, kernel_size};
    nvinfer1::IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 10, nvinfer1::DimsHW{4, 4}, wt, nvinfer1::Weights{});
    conv1->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv1->getOutput(0)->setName("fconv_cls_kernel");
    network->markOutput(*conv1->getOutput(0));
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,1 << 24);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );
    context.reset(engine->createExecutionContext());
    delete[] kernel_cls_wt;
}