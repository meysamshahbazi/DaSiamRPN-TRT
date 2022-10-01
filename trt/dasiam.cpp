#include "dasiam.hpp"



DaSiam::DaSiam()
{
    parseOnnxModel(temple_path,1U<<24,engine_temple,context_temple);
    parseOnnxModel(siam_path,1U<<24,engine_siam,context_siam);
    buffers_temple.reserve(engine_temple->getNbBindings());
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_temple->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_temple->getBindingDimensions(i)) * 1 * sizeof(float);
        cudaMalloc(&buffers_temple[i], binding_size);
        std::cout<<engine_temple->getBindingName(i)<<std::endl;
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
        std::cout<<engine_siam->getBindingName(i)<<std::endl;
        if (engine_siam->bindingIsInput(i))
        {  
            input_dims_siam.emplace_back(engine_siam->getBindingDimensions(i));
        }
        else
        {
            output_dims_siam.emplace_back(engine_siam->getBindingDimensions(i));
        }
    }
}

DaSiam::~DaSiam()
{
    for (void * buf : buffers_temple)
        cudaFree(buf);
    for (void * buf : buffers_siam)
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
    cudaMemcpyAsync(buffers_siam[0], blob, 3 * instance_size * instance_size * sizeof(float), cudaMemcpyHostToDevice);
    context_temple->enqueueV2(buffers_temple.data(), 0, nullptr);

    cv::waitKey(0);
    return Rect2f(0,0,100,100);// dummy return just for test
}