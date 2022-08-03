import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib as Path
import random
import math

def do_GaussianBlur(gray, kernel):
    blur_gray = cv2.GaussianBlur(gray,(kernel, kernel), 0)
    return blur_gray

def calbrightness(thimg):
    touch_botten = False
    thimg = cv2.cvtColor(thimg, cv2.COLOR_GRAY2RGB)
    thimg = cv2.cvtColor(thimg, cv2.COLOR_RGB2HSV)
    for val in range(0, thimg.shape[0]):
        now_y = thimg.shape[0]-val-1
        mean_bright = thimg[now_y, :, 2].mean()
        if (mean_bright > 100) & (touch_botten == False):
            # printtxt = "{} : {}".format(now_y,mean_bright)
            down_px = now_y
            # print(printtxt)
            touch_botten = True

        if touch_botten == True:
            if mean_bright < 100:
                # printtxt = "{} : {}".format(now_y,mean_bright)
                up_px=now_y
                # print(printtxt)
                return down_px,up_px

ALLOW_FORMAT = ["jpg", "jpge", "png", "bmp"]
path = "/home/nvidia/Desktop/findfinger/NG/original"
img_list = [path +"/"+ img for img in os.listdir(path) if img.split(".")[-1] in ALLOW_FORMAT]
#先以 "." 分開檔名,並透過 ALLOW_FORMAT 判斷是否為支援的格式
#再來 os.listdir(path) 的功能為列印出 path 下的檔案名稱
#最後加上path就是 儲存完整的路徑名稱的矩陣

defect = cv2.imread("defect.jpg")
for img in img_list:
    img = cv2.imread(img)
    # if img == None:
    #     continue
    h,w,_ = img.shape
    action_img  = img.copy()
    # action_img = action_img[h//2:,:]
    action_img_gray = cv2.cvtColor(action_img, cv2.COLOR_BGR2GRAY)
    blur_img = do_GaussianBlur(action_img_gray, 5)
    ret3,thimg = cv2.threshold(blur_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Otsu's Threshold 大津二值化

    down_px,up_px = calbrightness(thimg)
    action_img = action_img[up_px- 30 : down_px+ 30, :]# 取得金手指區域(彩色原圖) ＝ action_img 
    cut_img = thimg[up_px- 30 : down_px+ 30, :]# 取得金手指區域(灰階圖) ＝ cut_img 

    for cut_idx in range (0, 7):
        bla_background = np.zeros((608, 608, 3), np.uint8)
        split_oriimg= action_img[:,( cut_idx * 304 ):((cut_idx + 2)* 304)]
        split_grayimg= cut_img[:,( cut_idx * 304 ):((cut_idx + 2)* 304)]

        # Contours
        contours, hiera = cv2.findContours(split_grayimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # block = np.zeros(action_img.shape,"uint8")
        finger_cnt = []    #儲存全部 defect 可複製區域的中心點 cy,cx
        
        # Calculate area
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 1000:
                M = cv2.moments(cnt)
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                #上面三行計算圖片矩,透過 m10/m00 的方式 求得圖片中心位置
                finger_cnt.append([cy,cx])

        h,w,_= defect.shape
        hf_h,hf_w = math.ceil(h/2), math.ceil(w/2)
        final_img=split_oriimg.copy()

        for val in range(0,10):
            sel_val = random.randint(0,len(finger_cnt)-1)
            cy, cx = finger_cnt[sel_val]
            cy += random.randint(-30,30)
            final_img[cy- h//2: cy+hf_h, cx- w//2: cx+ hf_w] = defect
                
        # cv2.imshow("action_img", final_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        final_img_h,_,_=final_img.shape
        # print(cut_img_h)
        bla_background[ 200 : 200+ final_img_h,:,:]=final_img #create a black image
        cv2.imshow("out_img", bla_background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # cv2.imshow("gray_img", action_img)
    # cv2.imshow("blur_img", blur_img)
    # cv2.imshow("thershold_img", thimg)
    # cv2.imshow("cut_img", cut_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

