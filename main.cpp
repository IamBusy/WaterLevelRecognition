#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <dirent.h>

#include "helper.hpp"
#include "process.hpp"
#include "detect.hpp"

using namespace std;
using namespace cv;


#define TRAIN false     //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define CENTRAL_CROP true   //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体

//需要处理的目标文件
#define TARGET "target.jpg"
#define TARGET_PATH string("./targets/")

//数据来源
#define BASE_PATH string("./")
#define TRAIN_POS_PATH  BASE_PATH + string("pos_all/")
#define TRAIN_NEG_PATH  BASE_PATH + string("neg/")
#define TRAIN_HARD_PATH BASE_PATH + string("hard/")
#define SVM_PATH BASE_PATH

#define PRODUCTION 1
#define DEVELOPMENT 0
#define env PRODUCTION


int main()
{
    
    int mode=1;
    int top = 100;
    int ESize=1;
    
    if(env == PRODUCTION)
    {
        cout<<"请输入处理模式： (1:高清 2:模糊)"<<endl;
        cin>>mode;
        cout<<"请输入高度："<<endl;
        cin>>top;
        cout<<"请输入E大小："<<endl;
        cin>>ESize;
    }
    
    

        
    
    MySVM svm;
    //训练分类器
    if(TRAIN)
    {
        train(svm,TRAIN_POS_PATH,TRAIN_NEG_PATH,SVM_PATH+"svm.xml");
        
    }
    else
    {
        svm.load((SVM_PATH+"svm.xml").c_str());
    }
    
    

    
    HOGDescriptor *myHOG = detect(svm);
    
    

    
    //处理目标文件
    vector<string> targets = getAllFiles(TARGET_PATH);
    for (int i=0; i<targets.size(); i++) {
        
        
        Mat origin = imread(targets[i]);
        

        
        Mat mask;
        vector<Mat> masks;
        masks.push_back(filterColor(origin));
        masks.push_back(filterCanny(origin));
        
        mask = mergeMasks(masks);
        
        
        
        if(mode==1)
        {
            //腐蚀、膨胀
            int erosion_size = 3;
            Mat element = getStructuringElement( MORPH_RECT,
                                                Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                                Point( erosion_size, erosion_size ) );
            /// 腐蚀操作
            erode( origin, origin, element );
            dilate(origin, origin, element);
        }
        else if(mode == 2)
        {
            //创建并初始化滤波模板
            cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
            kernel.at<float>(1,1) = 5.0;
            kernel.at<float>(0,1) = -1.0;
            kernel.at<float>(1,0) = -1.0;
            kernel.at<float>(1,2) = -1.0;
            kernel.at<float>(2,1) = -1.0;
            cv::filter2D(origin,origin,origin.depth(),kernel);
            
            int alpha = 1.5;
            int beta = 50;
            for( int y = 0; y < origin.rows; y++ )
            {
                for( int x = 0; x < origin.cols; x++ )
                {
                    for( int c = 0; c < 3; c++ )
                    {
                        origin.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( origin.at<Vec3b>(y,x)[c] ) + beta );
                    }
                }
            }
            
        }
        
        
        
        Mat src = origin;
        
        cvtColor(src, src, CV_RGB2GRAY);
        
        equalizeHist( src, src );
        
        vector<Rect> found, found_filtered;//矩形框数组
        

        myHOG->detectMultiScale(src, found, 0, Size(8,8), Size(16,16), 1.05, 2);
        
   
        found = filterRect(mask, found);
        found = filterSinglePeak(found);
        
        
        int finalHeight=0;
        finalHeight = adjustRect(found);
        finalHeight = fitting(finalHeight);
        
 
        
        for (size_t i = 0; i < found.size(); i++)
        {
            cv::rectangle(origin, found[i], cv::Scalar(0, 255, 0),2);
        }
        
        imwrite(BASE_PATH+"processed/"+randName()+".png", origin);
    }
    

    return 0;
}
