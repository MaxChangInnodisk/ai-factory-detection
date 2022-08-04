import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from data_aug.data_aug import *
from data_aug.bbox_util import *

img_source = "/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/b_dot/b_dot_false.jpg"
img_template = "/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/false_small_defect/b_dot.jpg"
txt_source = "/home/nvidia/Desktop/findfinger/NG/tar_defect (flase)/b_dot/b_dot_false.txt"

def aug(img_source, txt_source):
    img=cv2.imread(img_source)  #read image       
    h,w,_=img.shape

    with open(txt_source, 'r') as f:
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
    return img_

aug_img = aug(img_source, txt_source)
cv2.imshow("aug_img", aug_img)


# tar_img = cv2.imread(aug_img)
tar_img = aug_img.copy()
tar_gray = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)
tar_defect = cv2.imread(img_template,0)
w, h = tar_defect.shape[::-1]

#
res = cv2.matchTemplate(tar_gray,tar_defect,cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]): 
    #[::-1]change y location to x location 
    #now pt return a (x,y) location which is >= threshold
    #与 zip 相反，*zipped 可理解为解压，返回二维矩阵
    cv2.rectangle(tar_img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imshow('res.png',tar_img)

cv2.waitKey(0)
cv2.destroyAllWindows