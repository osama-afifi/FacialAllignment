

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

string window_name = "Capture - Eye Iris detection";
RNG rng(12345);

void Display( const vector<Rect> &rect, Mat &frame)
{
	for( size_t i = 0; i < rect.size(); i++ )
    {
		Point center( rect[i].x + rect[i].width/2, rect[i].y + rect[i].height/2 );
		rectangle(frame,rect[i],Scalar(255,0,0));
	  //ellipse( frame, center, Size( rect[i].width/2, rect[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
    }
   imshow( window_name, frame );
}

void Display( const vector< pair<Point2i,Point2i> > &iris, Mat &frame)
{
	for( size_t i = 0; i < iris.size(); i++ )
    {
		circle( frame, iris[i].first , 5, Scalar(255,255,0));
		circle( frame, iris[i].second, 5, Scalar(255,255,0));
    }
   imshow( window_name, frame );
}


int main( void )
{
  VideoCapture capture;
  Mat frame;



  capture.open( 0 );
  if( capture.isOpened() )
  {
    while(true)
    {
      capture >> frame;
      //Apply the classifier to the frame
	  	vector<Rect> profilefaceRect;
	  if( !frame.empty() )  
	  {
		  Display(profilefaceRect, frame);
		  //imshow( window_name, frame );
	  }
      else { printf(" --(!) No captured frame -- Break!"); break; }
      int c = waitKey(10);
      if( (char)c == 'c') { break; }
	  if( (char)c == 's')
	  {
		imwrite( "test.jpg", frame );
	  }

    }
  }
  return 0;
}
