#include "dasiam.hpp"



DaSiam::DaSiam()
{

}

DaSiam::~DaSiam()
{

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
    return;
}

Rect2f DaSiam::update(const Mat &im)
{

    return Rect2f(0,0,100,100);// dummy return just for test
}