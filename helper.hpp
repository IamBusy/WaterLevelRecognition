//
//  helper.hpp
//  final_assignment
//
//  Created by william wei on 17/1/7.
//  Copyright © 2017年 simon. All rights reserved.
//

#ifndef helper_hpp
#define helper_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>




std::vector<std::string> getAllFiles(std::string cate_dir);


int random(double start, double end);



std::string randName();



//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public CvSVM
{
public:
    //获得SVM的决策函数中的alpha数组
    double * get_alpha_vector()
    {
        return this->decision_func->alpha;
    }
    
    //获得SVM的决策函数中的rho参数,即偏移量
    float get_rho()
    {
        return this->decision_func->rho;
    }
};



#endif /* helper_hpp */
