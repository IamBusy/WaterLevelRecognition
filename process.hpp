//
//  process.hpp
//  final_assignment
//
//  Created by william wei on 17/1/7.
//  Copyright © 2017年 simon. All rights reserved.
//

#ifndef process_hpp
#define process_hpp

#include <stdio.h>


#include <unistd.h>
#include <dirent.h>
#include<stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include "helper.hpp"

using namespace std;
using namespace cv;


#define COLOR_THRESHOLD 255
#define AREA_THRESHOLD 0.3
#define RECT_WIDTH_LIMIT 150
#define RECT_AREA 0.1



void overSpread(Mat &mask,int width=3,int height=6,int threshold = 200);


Mat filterColor(const Mat &inputImage);
Mat filterCanny(const Mat inputImage);

Mat mergeMasks(vector<Mat> Vmat);



vector<Rect> filterRect(const Mat inputImage,vector<Rect>found);
vector<Rect> filterSinglePeak(vector<Rect> R);
int adjustRect(vector<Rect> &vecrec);

int fitting(float input);

#endif /* process_hpp */
