#include <opencv2/highgui.hpp>
#include <iostream>
#include "dasiam.hpp"

#include "d_utils.h"

using namespace std;
using namespace cv;

int main(int argc, const char ** argv) 
{
    // set input video

// anchor = generate_anchor(total_stride=8, scales= [8, ], 
//                          ratios = [0.33, 0.5, 1, 2, 3], score_size =int(score_size))

    

    Rect2f roi = Rect(550.0f, 223.0f, 215.0f, 272.0f); // xywh format  
    Mat frame;
    // 000087.jpg
    std::string video {"/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/%06d.jpg"};//= argv[1];
    VideoCapture cap(video);

    // get bounding box
    cap >> frame;
    int64 tick_counter = 0;
    int frame_idx = 0;
    DaSiam ds;
    // return 0;
    frame_idx++;
    int64 t1 = cv::getTickCount();
    ds.init(frame,roi);
    int64 t2 = cv::getTickCount();
    tick_counter += t2 - t1;
    rectangle(frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
    
    cout << "FPSi: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;
    imshow("tracker",frame);
    waitKey(1);

    for ( ;; )
    {
        // get frame from the video
        cap >> frame;
        // cv::Mat temp;
        
        // cv::cvtColor(frame,temp,COLOR_BGR2YUV_I420);
        // stop the program if no more images
        if(frame.rows==0 || frame.cols==0)
            break;
        frame_idx ++;
        int64 t1 = cv::getTickCount();
        roi = ds.update(frame);
        int64 t2 = cv::getTickCount();
        tick_counter += t2 - t1;
        // std::cout<<roi.x<<" "<<roi.y<<" "<<roi.width<<" "<<roi.height<<endl;
        // Rect roi_int = 
        cout << "FPS: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;
        rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
        //  break;
        imshow("tracker",frame);
        
        if(waitKey(1)==27) break;
    }    

    return 0;
}




