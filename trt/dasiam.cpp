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

    return;
}

Rect2f DaSiam::update(const Mat &im)
{

    return Rect2f(0,0,100,100);// dummy return just for test
}