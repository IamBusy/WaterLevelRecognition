//
//  detect.hpp
//  final_assignment
//
//  Created by william wei on 17/2/22.
//  Copyright © 2017年 simon. All rights reserved.
//

#ifndef detect_hpp
#define detect_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "helper.hpp"


//识别与检测的参数
#define WIN_SIZE Size(64,64)
#define BLOCK_SIZE Size(8,8)
#define BLOCK_STRIDE Size(4,4)
#define CELL_SIZE Size(4,4)
#define BIN 12


void train(MySVM &svm,std::string posPath,std::string negPath,std::string savePath);
cv::HOGDescriptor* detect(MySVM &svm);

#endif /* detect_hpp */
