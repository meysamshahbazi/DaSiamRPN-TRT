#include "dasiam.hpp"

// #define GEN_ENGINE_FROM_ONNX
// #define LOAD_FROM_ONNX
#define LOAD_FROM_ENGINE

DaSiam::DaSiam()
{
#ifdef GEN_ENGINE_FROM_ONNX
    // saveEngineFile(temple_path,temple_path_engine);
    // saveEngineFile(siam_path,siam_path_engine);
    // saveEngineFile(regress_path,regress_path_engine);
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
    parseEngineModel(siam_path_engine,engine_siam,context_siam);
    parseEngineModel(regress_path_engine,engine_regress,context_regress);
#endif    

    buffers_temple.reserve(engine_temple->getNbBindings());
    // cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_temple->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_temple->getBindingDimensions(i)) * 1 * sizeof(float);
        
        // std::cout<<engine_temple->getBindingName(i)<<std::endl;
        // printDim(engine_temple->getBindingDimensions(i));
        if (engine_temple->bindingIsInput(i))
        {  
            input_dims_temple.emplace_back(engine_temple->getBindingDimensions(i));
        }
        else
        {
            cudaMalloc(&buffers_temple[i], binding_size);
            output_dims_temple.emplace_back(engine_temple->getBindingDimensions(i));
        }
    }
    
    buffers_siam.reserve(engine_siam->getNbBindings());
    // cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_siam->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_siam->getBindingDimensions(i)) * 1 * sizeof(float);
        
        // std::cout<<engine_siam->getBindingName(i);
        // printDim(engine_siam->getBindingDimensions(i));
        if (engine_siam->bindingIsInput(i))
        {  
            input_dims_siam.emplace_back(engine_siam->getBindingDimensions(i));
        }
        else
        {
            cudaMalloc(&buffers_siam[i], binding_size);
            output_dims_siam.emplace_back(engine_siam->getBindingDimensions(i));
        }
    }

    buffers_regress.reserve(engine_regress->getNbBindings());
    // cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_regress->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_regress->getBindingDimensions(i)) * 1 * sizeof(float);
        cudaMalloc(&buffers_regress[i], binding_size);
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
    binding_size = 10*19*19*sizeof(float);
    cudaMalloc(&d_score, binding_size);

    anchor = generate_anchor(total_stride,scales,ratios,score_size);
    float * h_anchor = new float[anchor.size()*4];

    // float* d_anchor;
    cudaMalloc(&d_anchor, anchor.size()*4*sizeof(float));

    for (int j=0; j<4; j++)
    {
        for(int i=0; i<anchor.size(); i++)
            h_anchor[j*anchor.size()+i] = anchor[i][j];
    }

    cudaMemcpy(d_anchor,h_anchor,anchor.size()*4*sizeof(float),cudaMemcpyHostToDevice);
    window = get_hann_win(Size(score_size,score_size));

    // window.at<float>(index_window/score_size,index_window%score_size)
    cout<<"anchor.size() "<<anchor.size()<<endl;
    float* h_window = new float[anchor.size()];

    int window_idx = 0;
    for(int i = 0;i<5;i++)
        for(int j = 0;j<score_size;j++)
            for(int k = 0;k<score_size;k++)
            {
                h_window[window_idx] = window.at<float>(j,k);
                window_idx++;
            }

    cudaMalloc(&d_window, anchor.size()*sizeof(float));
    cudaMemcpy(d_window,h_window,anchor.size()*sizeof(float),cudaMemcpyHostToDevice);


#ifdef CLASSIC_MEM_

    delta = new float[4*anchor.size()];
    score = new float[2*anchor.size()];
    blob = new float[3*instance_size*instance_size];
    temple_blob = new float[3*exemplar_size*exemplar_size];
    cudaMalloc(&buffers_siam[0], 3*instance_size*instance_size*sizeof(float));
    cudaMalloc(&buffers_temple[0], 3*exemplar_size*exemplar_size*sizeof(float));
#endif

#ifdef UNIFIED_MEM_

    cudaMallocManaged((void **)&blob,3*instance_size*instance_size*sizeof(float),cudaMemAttachHost);
    cudaMallocManaged((void **)&temple_blob,3*exemplar_size*exemplar_size*sizeof(float),cudaMemAttachHost);
    buffers_siam[0] = (void *) blob;
    buffers_temple[0] = (void *) temple_blob;

#endif


    // CUDNN stuff 
    checkCUDNN(cudnnCreate(&cudnn));
    checkCUDNN(cudnnCreateTensorDescriptor(&cls1_in_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(cls1_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 22, 22));
    checkCUDNN(cudnnCreateTensorDescriptor(&r1_in_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(r1_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 22, 22));
    checkCUDNN(cudnnCreateTensorDescriptor(&cls1_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(cls1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 10, 19, 19));
    checkCUDNN(cudnnCreateTensorDescriptor(&r1_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(r1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 20, 19, 19));
    checkCUDNN(cudnnCreateFilterDescriptor(&cls1_kernel_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(cls1_kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 10, 256, 4, 4));
    checkCUDNN(cudnnCreateFilterDescriptor(&r1_kernel_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(r1_kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 20, 256, 4, 4));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&cls1_conv_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(cls1_conv_desc, 0, 0, 1, 1, 1, 1,
                                            /*mode=*/CUDNN_CROSS_CORRELATION, // TODO: it may need change ... 
                                            /*computeType=*/CUDNN_DATA_FLOAT));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&r1_conv_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(r1_conv_desc, 0, 0, 1, 1, 1, 1,
                                            /*mode=*/CUDNN_CROSS_CORRELATION, // TODO: it may need change ... 
                                            /*computeType=*/CUDNN_DATA_FLOAT));

    cudnnGetConvolutionForwardWorkspaceSize(cudnn, cls1_in_desc,cls1_kernel_desc,cls1_conv_desc,
                                            cls1_out_desc,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                            &cls1_workspace_bytes);

    cudnnGetConvolutionForwardWorkspaceSize(cudnn, r1_in_desc, r1_kernel_desc, r1_conv_desc,
                                            r1_out_desc,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                            &r1_workspace_bytes);

    cudaMalloc(&cls1_d_workspace, cls1_workspace_bytes);
    cudaMalloc(&r1_d_workspace, r1_workspace_bytes);

    // ----------------------------------------------------------------
    // warm-up
    for(int i=0;i<10;i++)
    {
        context_temple->enqueueV2(buffers_temple.data(), 0, nullptr);
        context_siam->enqueueV2(buffers_siam.data(), 0, nullptr);
        checkCUDNN(
        cudnnConvolutionForward(cudnn, &cudnn_alpha, cls1_in_desc, buffers_siam[2], cls1_kernel_desc, buffers_temple[2],
                            cls1_conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, cls1_d_workspace, cls1_workspace_bytes, 
                            &cudnn_beta, cls1_out_desc, d_score) ); // TODO change buffer buffers_cls[1]

        checkCUDNN(
        cudnnConvolutionForward(cudnn, &cudnn_alpha, r1_in_desc, buffers_siam[1], r1_kernel_desc, buffers_temple[1],
                            r1_conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, r1_d_workspace, r1_workspace_bytes, 
                            &cudnn_beta, r1_out_desc, buffers_regress[0]) ); 
        context_regress->enqueueV2(buffers_regress.data(), 0, nullptr);
        cudaStreamSynchronize(0);
    }
    
}

DaSiam::~DaSiam()
{
    for (void * buf : buffers_temple)
        cudaFree(buf);
    for (void * buf : buffers_siam)
        cudaFree(buf);

    for (void * buf : buffers_regress)
        cudaFree(buf);

    cudaFree(d_score);
    delete[] score;
    delete[] delta;
    delete[] blob;
    delete[] temple_blob;
}

void DaSiam::init(const Mat &im, const Rect2f state)
{
    // __add_da();
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
    // cv::cvtColor(z_crop,z_crop,COLOR_YUV2BGR_I420);
    blobFromImage(z_crop,temple_blob);
#ifdef CLASSIC_MEM_
    cudaMemcpyAsync(buffers_temple[0], temple_blob, 3 * exemplar_size * exemplar_size * sizeof(float), cudaMemcpyHostToDevice);
#endif
#ifdef UNIFIED_MEM_
    cudaStreamAttachMemAsync(0,temple_blob,0,cudaMemAttachGlobal);
#endif
    context_temple->enqueueV2(buffers_temple.data(), 0, nullptr);
    // cudaStreamSynchronize(0); // this is vital to wait for last results ..
    // create_fconv_r(engine_r1,context_r1);
    // create_fconv_cls(engine_cls,context_cls);

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
    // cv::cvtColor(x_crop,x_crop,COLOR_YUV2BGR_I420);
    // // imshow("tracker",x_crop);
    // // waitKey(0);
    target_sz = target_sz*scale_z;
    blobFromImage(x_crop,blob);
#ifdef CLASSIC_MEM_
    cudaMemcpyAsync(buffers_siam[0], blob, 3 * instance_size * instance_size * sizeof(float), cudaMemcpyHostToDevice);
#endif
#ifdef UNIFIED_MEM_
    cudaStreamAttachMemAsync(0,blob,0,cudaMemAttachGlobal);
#endif

    context_siam->enqueueV2(buffers_siam.data(), 0, nullptr);
    checkCUDNN(
    cudnnConvolutionForward(cudnn, &cudnn_alpha, cls1_in_desc, buffers_siam[2], cls1_kernel_desc, buffers_temple[2],
                            cls1_conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, cls1_d_workspace, cls1_workspace_bytes, 
                            &cudnn_beta, cls1_out_desc, d_score) ); // TODO change buffer buffers_cls[1]

    checkCUDNN(
    cudnnConvolutionForward(cudnn, &cudnn_alpha, r1_in_desc, buffers_siam[1], r1_kernel_desc, buffers_temple[1],
                            r1_conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, r1_d_workspace, r1_workspace_bytes, 
                            &cudnn_beta, r1_out_desc, buffers_regress[0]) ); 


    // buffers_regress[0] = buffers_r1[1];
    context_regress->enqueueV2(buffers_regress.data(), 0, nullptr);
    
    #define d_delta buffers_regress[1] // device pointer that represnts delta array
    int delta_size = anchor.size()*4;
    

    
    
    
    //----------------------------------------------------------
    // #define CPU_POST_PROCESS


    #ifdef CPU_POST_PROCESS

    cudaMemcpyAsync(delta, buffers_regress[1], delta_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(score, d_score, 2*anchor.size()*sizeof(float), cudaMemcpyDeviceToHost);    
    cudaStreamSynchronize(0);
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
                    sz(delta[i + 2*anchor.size()], delta[i + 3*anchor.size()])/
                    sz(target_sz.width, target_sz.height)
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
    #endif

    //---------------------------------------------------------------------------------------------------

    #define GPU_POST_PROCESS
    
    #ifdef GPU_POST_PROCESS

    float * ret = new float[6];
    // cudaStreamSynchronize(0);
    post_process(static_cast<float*>(d_delta),d_anchor, d_score,d_window,
        window_influence,target_sz.width,target_sz.height,penalty_k,anchor.size(),
        ret,0);
    
    float res_x = ret[0]/scale_z;
    float res_y = ret[1]/scale_z;
    float res_w = ret[2]/scale_z;
    float res_h = ret[3]/scale_z;
    float lr = ret[4]*ret[5]*p_lr;
    #endif

  
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
    // config->setMaxWorkspaceSize(1U<<24);
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
    // config->setMaxWorkspaceSize(1U<<24);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );
    context.reset(engine->createExecutionContext());
    delete[] kernel_cls_wt;
}