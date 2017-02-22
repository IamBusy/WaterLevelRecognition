//
//  detect.cpp
//  final_assignment
//
//  Created by william wei on 17/2/22.
//  Copyright © 2017年 simon. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include "detect.hpp"
#include "helper.hpp"
#include "process.hpp"

using namespace cv;

HOGDescriptor* detect(MySVM &svm)
{
    HOGDescriptor hog(WIN_SIZE,BLOCK_SIZE,BLOCK_STRIDE,CELL_SIZE,BIN);  //HOG检测器，用来计算HOG描述子的
    int DescriptorDim;//HOG描述子的维数

    
    DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
    int supportVectorNum = svm.get_support_vector_count();//支持向量的个数

    
    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
    Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
    Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果
    
    //将支持向量的数据复制到supportVectorMat矩阵中
    for(int i=0; i<supportVectorNum; i++)
    {
        const float * pSVData = svm.get_support_vector(i);
        for(int j=0; j<DescriptorDim; j++)
        {
            supportVectorMat.at<float>(i,j) = pSVData[j];
        }
    }
    
  
    //返回SVM的决策函数中的alpha向量
    double * pAlphaData = svm.get_alpha_vector();
    for(int i=0; i<supportVectorNum; i++)
    {
        alphaMat.at<float>(0,i) = pAlphaData[i];
    }

    resultMat = -1 * alphaMat * supportVectorMat;
    

    vector<float> myDetector;
    
    //将resultMat中的数据复制到数组myDetector中
    for(int i=0; i<DescriptorDim; i++)
    {
        myDetector.push_back(resultMat.at<float>(0,i));
    }

    myDetector.push_back(svm.get_rho());
    
    HOGDescriptor *myHOG = new HOGDescriptor(WIN_SIZE,BLOCK_SIZE,BLOCK_STRIDE,CELL_SIZE,BIN);

    myHOG->setSVMDetector(myDetector);
    return myHOG;
}


void train(MySVM &svm,string posPath,string negPath,string savePath="")
{
    HOGDescriptor hog(WIN_SIZE,BLOCK_SIZE,BLOCK_STRIDE,CELL_SIZE,BIN);//HOG检测器，用来计算HOG描述子的
    unsigned int DescriptorDim = 0;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
    vector<string> posImg = getAllFiles(posPath);
    vector<string> negImg = getAllFiles(negPath);
    unsigned long posNum = posImg.size();
    unsigned long negNum = negImg.size()*2;
    string ImgName;//图片名(绝对路径)
    
    
    Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
    Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人
    
    for (int num=0; num<posImg.size(); num++)
    {
        Mat origin = imread(posImg[num]);
        Mat src;
        
        std::cout<<posImg[num];
        
        vector<float> descriptors;//HOG描述子向量
        hog.compute(src,descriptors,BLOCK_STRIDE);//计算HOG描述子，检测窗口移动步长(8,8)
        
        //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
        if( num == 0 )
        {
            DescriptorDim = descriptors.size();//HOG描述子的维数

            sampleFeatureMat = Mat::zeros(posNum+negNum, DescriptorDim, CV_32FC1);
            sampleLabelMat = Mat::zeros(posNum+negNum, 1, CV_32FC1);
        }
        
        //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
        for(int i=0; i<DescriptorDim; i++)
            sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
        
        sampleLabelMat.at<float>(num,0) = 1;//正样本类别为1，有人
    }
    
    
    
    //依次读取负样本图片，生成HOG描述子
    for(int j=0; j<negNum; j++)
    {
        int num = j/2;
        
        Mat origin = imread(negImg[num]);
        Mat src;
        cvtColor( origin, src, CV_BGR2GRAY );
        Mat img1 = src(Rect(0,0,64,64));
        Mat img2 = src(Rect(0,64,64,64));
        
        vector<float> descriptors;//HOG描述子向量
        hog.compute(img1,descriptors,BLOCK_STRIDE);//计算HOG描述子，检测窗口移动步长(8,8)

        
        //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
        for(int i=0; i<DescriptorDim; i++)
            sampleFeatureMat.at<float>(j+posNum,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
        
        sampleLabelMat.at<float>(j+posNum,0) = -1;//负样本类别为-1，无人
    }
    
    
    //训练SVM分类器
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

    CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
    
    svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
    if(savePath.length()>0)
    {
        svm.save(savePath.c_str());//将训练好的SVM模型保存为xml文件
    }
    
    
}


