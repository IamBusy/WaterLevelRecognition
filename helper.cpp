//
//  helper.cpp
//  final_assignment
//
//  Created by william wei on 17/1/7.
//  Copyright © 2017年 simon. All rights reserved.
//

#include "helper.hpp"
#include <vector>
#include <string>

#include <unistd.h>
#include <dirent.h>
#include<stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;


vector<string> getAllFiles(string cate_dir)
{
    vector<string> files;//存放文件名
    
    DIR *dir;
    struct dirent *ptr;
    
    if ((dir=opendir(cate_dir.c_str())) == NULL)
    {
        perror("Open dir error...");
    }
    
    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        
        else if(ptr->d_type == 8){    ///file
            if(string(ptr->d_name).find(".jpg")!=string::npos or string(ptr->d_name).find(".png")!=string::npos)
            {
                files.push_back(cate_dir+ptr->d_name);
            }
        
            else
            {
                continue;
            }
        }
        else{
            vector<string> nextFiles = getAllFiles(cate_dir+ptr->d_name+"/");
            for (int i=0; i<nextFiles.size(); i++) {
                files.push_back(nextFiles[i]);
            }
        }
    }
    
    closedir(dir);
    //排序，按从小到大排序
    sort(files.begin(), files.end());
    return files;
}

int random(double start, double end)
{
    return start+(end-start)*rand()/(RAND_MAX + 1.0);
}


string randName()
{
    int length = random(3, 6);
    string result="";
    for (int i=0; i<length; i++)
    {
        char c = random(0,60)+60;
        result +=c;
    }
    return result;
}


