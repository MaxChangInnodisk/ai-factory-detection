from termios import CINTR
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib as Path
import random

from data_aug.data_aug import *
from data_aug.bbox_util import *

CONV_OUT_PATH='/home/nvidia/Desktop/findfinger/NG/ng_defect_AUG/'


TIMES=2
ALLOW_FORMAT = ["jpg", "jpge", "png", "bmp"]
countval=1

def aug(img_source, times ,countval):
    for i in range(1,times+1):
        countval +=1
        # print(i)
        # get all file
        for file_name in glob.glob(img_source + "/*.jpg"):
            file_name = os.path.splitext(file_name)[0]  #   remove ".txt"
            file_name = file_name.split('/')[-1]    #   get file name
            # print(file_name)#
            img_name = (img_source+file_name+".jpg")
            img=cv2.imread(img_name)  #read image       
            h,w,_=img.shape

            txt_name=(img_source+file_name+ ".txt")
            with open(txt_name, 'r') as f:
                temp=f.readlines()
                all_box = []
                for line in temp:
                    line=line.strip()
                    temp2=line.split(" ")   #strip()方法用於移除字符串頭尾指定的字符

                    #Save the paremeter in the Yolo text
                    x_,y_,w_,h_=eval(temp2[1]),eval(temp2[2]),eval(temp2[3]),eval(temp2[4]) 

                    x1=w*x_- 0.5* w* w_
                    x2=w*x_+ 0.5* w* w_
                    y1=h*y_- 0.5* h* h_
                    y2=h*y_+ 0.5* h* h_
                    bboxes=[x1, y1, x2, y2, 0]
                    all_box.append(bboxes)

            bboxes=np.array(all_box)
            
            seq = Sequence([RandomTranslate(0.3, diff = True),
                            RandomScale(0.2, diff = True),
                            # RandomHorizontalFlip()
                            ])

            img_, bboxes_ = seq(img.copy(), bboxes.copy())
            if len(bboxes_) != 0:
                #   change to Yolo txt
                bboxes_list=[]
                for val in bboxes_:
                    x1, y1, x2, y2 = val[0], val[1], val[2], val[3]
                    x_= (x1+ x2)/ (2* w)
                    y_= (y1+ y2)/ (2* h)
                    w_= (x2- x1)/ w
                    h_= (y2- y1)/ h
                    newline = "{} {} {} {} {}".format(
                                temp2[0],
                                x_, 
                                y_, 
                                w_, 
                                h_
                                )
                    bboxes_list.append(newline)

                #   crate a new txt file to save new bboxes
                txtpath=(CONV_OUT_PATH+ file_name+ "_aug"+ str(countval)+ ".txt")
                with open (txtpath, 'w') as f:
                    for item in bboxes_list:
                        f.write("%s\n" %item)

                #   output image
                outputImg_name=(CONV_OUT_PATH+ file_name+ "_aug"+ str(countval)+ ".jpg")
                cv2.imwrite(outputImg_name, img_)
    print(countval)            
    return countval


img_source ='/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/b_dot/'
countval=aug(img_source, 7*TIMES ,countval)

img_source ='/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/b_fuzzy/'
countval=aug(img_source, 3*TIMES ,countval)

img_source ='/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/b_scratch/'
countval=aug(img_source, 7*TIMES ,countval)

img_source ='/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/loss/'
countval= aug(img_source, 3*TIMES ,countval)

img_source ='/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/serios_scratch/'
countval=aug(img_source, 3*TIMES ,countval) 

img_source ='/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/si/'
countval=aug(img_source, 2*TIMES ,countval)

img_source ='/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/w_dot/'
countval=aug(img_source, 8*TIMES ,countval)

img_source ='/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/w_scratch/'
countval=aug(img_source, 3*TIMES ,countval)