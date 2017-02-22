//
//  process.cpp
//  final_assignment
//
//  Created by william wei on 17/1/7.
//  Copyright © 2017年 simon. All rights reserved.
//

#include "process.hpp"

#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "helper.hpp"

using namespace std;
using namespace cv;


/**
 * 颜色过滤，去除非蓝红以外的颜色
 *
 */
Mat filterColor(const Mat &inputImage)
{
    Mat hsvImage;
    cvtColor(inputImage, hsvImage, CV_BGR2HSV);
    Mat resultGray = Mat(hsvImage.rows, hsvImage.cols,CV_8U,cv::Scalar(255));
    //Mat resultColor = Mat(hsvImage.rows, hsvImage.cols,CV_8UC3,cv::Scalar(255, 255, 255));
    double H=0.0,S=0.0,V=0.0;
    for(int i=0;i<hsvImage.rows;i++)
    {
        for(int j=0;j<hsvImage.cols;j++)
        {
            H=hsvImage.at<Vec3b>(i,j)[0];
            S=hsvImage.at<Vec3b>(i,j)[1];
            V=hsvImage.at<Vec3b>(i,j)[2];
            
            if((S >= 20 && S<=255))
            {
                if(((  H>=0 && H <= 10) ||(H<=180&&H>=125) || (H>=100&&H<=124))&& V >= 0 && V <=255)
                {
                    resultGray.at<uchar>(i,j)=0;
                }
                
            }
        }
    }
    
    for (int i=0; i<10; i++) {
        resultGray = 255 - resultGray;
        overSpread(resultGray,10,10);
        resultGray = 255 - resultGray;
        overSpread(resultGray,10,10);
    }
    
    //resultGray = 255 - resultGray;
    overSpread(resultGray,20,50);
    
    
    return resultGray;
}



/**
 * 过滤异常大小的识别矩形
 *
 */
vector<Rect> filterRect(vector<Rect> R)
{
    vector<Rect>result;
    for (int i=0; i<R.size(); i++)
    {
        
        if(R[i].width<=RECT_WIDTH_LIMIT)
            result.push_back(R[i]);
    }
    return result;
}



vector<Rect> filterRect(const Mat inputImage,vector<Rect>found)
{
    Mat src = inputImage;
    
    vector<Rect> result;
    for (size_t i = 0; i < found.size(); i++)
    {
        if(found[i].width > RECT_WIDTH_LIMIT)
        {
            continue;
        }
        int pixel_sum=0;
        int x_coord=found[i].x;
        int y_coord=found[i].y;
        x_coord = x_coord<0?0:x_coord;
        y_coord = y_coord<0?0:y_coord;
        int width=found[i].width;
        int height=found[i].height;
        for(int x=x_coord; x<x_coord+width && x<inputImage.cols; x++)
        {
            for(int y=y_coord; y<y_coord+height && y<inputImage.rows; y++)
            {
                if(src.at<uchar>(y,x)<255)
                {
                    pixel_sum++;
                }
            }
            if (pixel_sum/(width*height*1.0)>AREA_THRESHOLD) {
                result.push_back(found[i]);
            }
        }
    }
    
    return result;
}


/**
 * 将过细的部分膨胀铺展开
 *
 */
void overSpread(Mat &mask,int width,int height,int threshold)
{
    int dilate_ele_size=30;
    
    Mat ele = getStructuringElement(MORPH_RECT, Size(width, height),
                                    Point(width/2, height/2));
    
    erode(mask, mask, ele);

}


/**
 * 单峰过滤
 *
 */
vector<Rect> filterSinglePeak(vector<Rect> R)
{
    if(R.size()==0)
        return R;
    
    int WD = 1.5;
    int Asize = R.size();
    int avew = 0; int aveh = 0; int rx = 0;int dy = 0;
    int lx = 9999;int uy = 9999;
    for (int i = 0; i < Asize; i++){
        avew = avew + R[i].width;
        aveh = aveh + R[i].height;
        if (lx>R[i].x)
            lx = R[i].x;
        if (uy > R[i].y)
            uy = R[i].y;
        if (rx < R[i].x + R[i].width)
            rx = R[i].x+R[i].width;
        if (dy < R[i].y + R[i].height)
            dy = R[i].y + R[i].height;
    }
    avew = avew / Asize;
    aveh = aveh / Asize;
    int choose, max = 0;
    int in = 0;
    for (int i = 0; i < Asize; i++){
        int tlx = R[i].x;
        for (int j = 0; j < Asize; j++){
            if (R[j].x >= tlx && R[j].x < tlx + WD * avew)
                in++;
        }
        if (in>max)
        {
            max = in;
            choose = i;
        }
        in = 0;
    }
    int tlx = R[choose].x;
    vector < Rect > result;
    for (int i = 0; i < Asize; i++){
        if (R[i].x >= tlx && R[i].x < tlx + WD * avew)
        {
            result.push_back(R[i]);
            //printf("%d %d %d ", R[i].x, tlx, tlx + 2 * avew);
        }
    }
    return result;
}



/**
 * 将各个mask进行叠加，得到最终的mask
 *
 */
Mat mergeMasks(vector<Mat> Vmat){
    int recl = 3;
    int all = recl*recl;
    int size = Vmat.size();
    int Rows= Vmat[0].rows;
    int Cols = Vmat[0].cols;
    double percent = 0.75;
    int temp = 0;
    int count = all*percent;
    bool control = true;
    Mat result(Rows, Cols,CV_8U,Scalar(255));
    for (int i = 0; i < Rows-recl; i = i + recl)
    {
        for (int j = 0; j < Cols-recl; j = j + recl)
        {
            
            for (int p = 0; p < recl; p++)
            {
                for (int q = 0; q < recl; q++)
                {
                    for (int k = 0; k < size; k++)
                    {
                        if (Vmat[k].at<uchar>(i + p, j + q) > 128)
                            control = false;
                    }
                    if (control == true)
                        temp++;
                    control = true;
                }
            }
            //printf("%d", temp);
            if (temp >= count){
                //printf("%d", temp);
                for (int t = 0; t < recl; t++)
                {
                    for (int s = 0; s < recl; s++){
                        //printf("%d", temp);
                        for (int z = 0; z < size; z++){
                            if (Vmat[z].at<uchar>(i + t, j + s) < 128)
                            {
                                result.at<uchar>(i + t, j + s) = 0;
                                //printf("%d", temp);
                            }
                        }
                    }
                }
            }
            temp = 0;
        }
    }
    
    return result;
}


/**
 * 基于Canny边缘检测的过滤
 *
 */
Mat filterCanny(const Mat inputImage)
{
    
    Mat origin = inputImage;
    
    GaussianBlur(origin, origin, Size(5, 5), 0, 0);

    
    Mat result = Mat(inputImage.rows, inputImage.cols,CV_8U,cv::Scalar(255));
    
    //blur(origin,origin,Size(3,3));
    Canny(origin, origin, 50, 100, 3);

    
    origin = 255 - origin;
    
    overSpread(origin,35,70);
    
    
    return origin;
}




bool CompRectY(Rect A,Rect B)
{
    if (A.y > B.y)
        return true;
    else
        return false;
}
bool CompRectX(Rect A, Rect B)
{
    if (A.x < B.x)
        return true;
    else
        return false;
}
vector<Rect> adjustOneRect(vector<Rect>vecrec,int ave)
{
    int Size = vecrec.size();
    vector<Rect>result;
    if (Size == 1)
        return vecrec;
    result.push_back(vecrec[0]);
    int iter = 0;
    int iter2 = Size - 1;
    double mul = 0;
    for (int i = 1; i < Size; i++){
        mul = (vecrec[iter].y - vecrec[i].y)*1.0 / (ave*1.0);
        if (int(mul + 0.5) % 2 == 1 || int(mul + 0.5)==1)
            continue;
        else
        {
            //iter = i;
            for (int j = 1; j <= (int(mul + 0.5))/2; j++)
                result.push_back(Rect((vecrec[iter].x + vecrec[i].x) / 2, vecrec[iter].y - 2*j*ave, ave, ave));
            iter = i;
        }
    }
    return result;
}
vector<Rect> adjustTwoRect(vector<Rect>vecrec, int ave, int leftx)
{
    int Size = vecrec.size();
    vector<Rect>result;
    if (Size == 1)
        return vecrec;
    result.push_back(vecrec[0]);
    int iter = 0;
    int iter2 = Size - 1;
    double mul = 0;
    bool left = false;
    if (vecrec[0].x > leftx + ave / 2)
        left = true;
    for (int i = 1; i < Size; i++){
        mul = (vecrec[iter].y - vecrec[i].y)*1.0 / (ave*1.0);
        if (int(mul + 0.5) == 1)
            continue;
        else
        {
            //iter = i;
            for (int j = 1; j <= (int(mul + 0.5)); j++)
            {
                if (left == false)
                    result.push_back(Rect(leftx+ave, vecrec[iter].y - j*ave, ave, ave));
                else
                    result.push_back(Rect(leftx, vecrec[iter].y - j*ave, ave, ave));
                left = !left;
            }
            iter = i;
        }
    }
    return result;
}

/**
 *  对识别结果进行调整补齐
 *
 */
int adjustRect(vector<Rect> &vecrec)
{
    int mode = 1;
    int Size = vecrec.size();
    int ave=0;
    for (int i = 0; i < Size; i++)
        ave = ave + vecrec[i].height;
    if(Size==0)
        return 0;
    ave = ave / Size;
    //printf("%d", ave);
    sort(vecrec.begin(), vecrec.end(), CompRectX);
    //printf("%d \n", vecrec[vecrec.size() - 1].x - vecrec[0].x);
    if (vecrec[vecrec.size() - 1].x - vecrec[0].x > ave / 2)
        mode = 2;
    //printf("%d", mode);
    if (mode == 1){
        sort(vecrec.begin(), vecrec.end(), CompRectY);
        int iter = 0;
        int iter2 = Size - 1;
        double mul = 0;
        for (int i = 0; i < Size - 1; i++){
            mul = (vecrec[i].y - vecrec[i + 1].y)*1.0 / (ave*1.0);
            if (int(mul + 0.5) <= 3)
            {
                iter = i;
                break;
            }
        }
        for (int i = Size - 1; i <0; i--){
            mul = (vecrec[i - 1].y - vecrec[i].y)*1.0 / (ave*1.0);
            if (int(mul + 0.5) <= 3)
            {
                iter2 = i;
                break;
            }
        }
        vector<Rect> result;
        for (int i = iter; i < iter2; i++)
        {
            result.push_back(vecrec[i]);
        }
        result = adjustOneRect(result, ave);
        vecrec = result;
        return 2*vecrec.size();
    }
    if (mode == 2)
    {
        vector<Rect>left;
        vector<Rect>right;
        int leftx = vecrec[0].x;
        sort(vecrec.begin(), vecrec.end(), CompRectY);
        int iter = 0;
        int iter2 = Size - 1;
        double mul = 0;
        for (int i =0; i < Size-1; i++){
            mul = (vecrec[i].y - vecrec[i+1].y)*1.0 / (ave*1.0);
            if (int(mul + 0.5) <= 3)
            {
                iter = i;
                break;
            }
        }
        for (int i = Size-1; i <0; i--){
            mul = (vecrec[i-1].y - vecrec[i].y)*1.0 / (ave*1.0);
            if (int(mul + 0.5) <= 3)
            {
                iter2 = i;
                break;
            }
        }
        for (int i = iter; i < iter2; i++)
        {
            if (vecrec[i].x < leftx + ave / 2)
            {
                left.push_back(vecrec[i]);
                //printf("%d %d \n", left.size(),vecrec[i].x);
            }
            else
                right.push_back(vecrec[i]);
        }
        //printf("%d \n", left.size());
        left = adjustOneRect(left,ave);
        right = adjustOneRect(right,ave);
        //printf("%d %d \n", left.size(),right.size());
        //printf("%d \n", left.size());
        sort(left.begin(), left.end(), CompRectY);
        int sumE = (max(left[0].y, right[0].y) - min(left[left.size() - 1].y, right[right.size() - 1].y)) / ave;
        if (abs(left[0].y - right[0].y)> 1.5*ave||abs(left[left.size() - 1].y - right[right.size() - 1].y) > 1.5*ave){
            for (int i = 0; i < right.size(); i++)
                left.push_back(right[i]);
            left=adjustTwoRect(left, ave, leftx);
        }
        else{
            for (int i = 0; i < right.size(); i++)
                left.push_back(right[i]);
        }
        //printf("%d \n", left.size());
        vecrec = left;
        return sumE;
    }
    return 1;
};

/**
 * 线性拟合调整函数
 */
int fitting(float input)
{
    return (int)-0.0003*input*input*input+0.0308*input*input+0.6015*input+6.2104;
    //return input;
}
