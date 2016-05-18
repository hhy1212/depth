#include <stdio.h>
#include <iostream>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

const char *windowDisparity = "Disparity";

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  VideoCapture cap0(1), cap1(2);
  Mat frame0, frame1, frame0_gray, frame1_gray;
  Mat imgDisparity16S = Mat( 480, 640,/*frame0.rows, frame0.cols,*/ CV_16S );
  Mat imgDisparity8U = Mat( 480, 640,/*frame0.rows, frame0.cols,*/ CV_8UC1 );
  if(!cap0.isOpened()){
    printf("Error: could not load camera 0.\n");
    return -1;
  }
  if(!cap1.isOpened()){
    printf("Error: could not load camera 1.\n");
    return -1;
  }
  namedWindow("left");
  namedWindow("right");
  namedWindow( windowDisparity, WINDOW_NORMAL );
  while (1){
    waitKey(20);
    cap0 >> frame0;
    cap1 >> frame1;
    if(!frame0.data){
      printf("Error: no frame data from camera 0\n");
      break;
    }
    if(!frame1.data){
      printf("Error: no frame data from camera 1\n");
      break;
    }
    flip(frame0, frame0, 1);
    flip(frame1, frame1, 1);
    GaussianBlur(frame0,frame0,Size(5,5),0,0);
    GaussianBlur(frame1,frame1,Size(5,5),0,0);
   // medianBlur(frame0,frame0,3);
   // medianBlur(frame1,frame1,3);
   // bilateralFilter(frame0,frame0,5,10,0,2.0);
    //bilateralFilter(frame1,frame1,5,10,0,2.0);
    cvtColor(frame0, frame0_gray, CV_BGR2GRAY);
    equalizeHist(frame0_gray, frame0_gray);
    cvtColor(frame1, frame1_gray, CV_BGR2GRAY);
    equalizeHist(frame1_gray, frame1_gray);

/*  //-- 1. Read the images
  Mat imgLeft = imread( argv[1], IMREAD_GRAYSCALE );
  Mat imgRight = imread( argv[2], IMREAD_GRAYSCALE );*/
  //-- And create the image in which we will save our disparities

/*
  if( imgLeft.empty() || imgRight.empty() )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
*/
  //-- 2. Call the constructor for StereoBM
    int ndisparities = 16*5;   /**< Range of disparity */
    int SADWindowSize = 21; /**< Size of the block window. Must be odd */

    Ptr<StereoBM> sbm = StereoBM::create( ndisparities, SADWindowSize );

  //-- 3. Calculate the disparity image
    sbm->compute( frame0_gray, frame1_gray, imgDisparity16S );

  //-- Check its extreme values
    double minVal; double maxVal;

    minMaxLoc( imgDisparity16S, &minVal, &maxVal );

    printf("Min disp: %f Max value: %f \n", minVal, maxVal);

  //-- 4. Display it as a CV_8UC1 image
    imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
    imshow("left",frame0_gray);
    imshow("right",frame1_gray);
    imshow( windowDisparity, imgDisparity8U );
/*
  //-- 5. Save the image
  imwrite("SBM_sample.png", imgDisparity16S);
*/
    waitKey(20);
  }

    return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SBMSample <imgLeft> <imgRight>" << std::endl; }
