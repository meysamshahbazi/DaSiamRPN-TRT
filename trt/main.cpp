#include <opencv2/highgui.hpp>
#include <iostream>
#include "dasiam.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char ** argv) 
{
    // set input video
    Rect2f roi = Rect(550.0f, 223.0f, 215.0f, 272.0f); // xywh format  
    Mat frame;
    // 000087.jpg
    std::string video {"/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/%06d.jpg"};//= argv[1];
    VideoCapture cap(video);

    // get bounding box
    cap >> frame;
    DaSiam ds;
    ds.init(frame,roi);
    rectangle(frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("tracker",frame);
    // waitKey(0);
    int64 tick_counter = 0;
    int frame_idx = 1;
    for ( ;; )
    {
        // get frame from the video
        cap >> frame;
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
        imshow("tracker",frame);
        
        if(waitKey(1)==27) break;
    }    

    return 0;
}





