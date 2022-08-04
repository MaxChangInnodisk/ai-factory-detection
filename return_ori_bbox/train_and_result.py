import cv2
import glob
# import matplotlib.pyplot as plt
import numpy as np
import os

from data_aug.data_aug import *
from data_aug.bbox_util import *

'''
原圖>切割出金手指>切成數等份(保留切割時的位置資訊)
自動產生資料增強>進行訓練
得到inference結果>回推bbox
'''
H=608
W=608
# CONV_OUT_PATH = "/home/nvidia/Desktop/Intern_project/return_ori_img/cut_Test_img(03)/"
CONV_OUT_PATH = "/workspace/cut_Test_img(03)/"

# img_name = '/home/nvidia/Desktop/Intern_project/return_ori_img/Test_img/01.jpg'
# img = cv2.imread(img_name)
blcbackgroung = np.zeros((H,W,3), np.uint8)

# path = "/home/nvidia/Desktop/Intern_project/return_ori_img/Test_img"
# txtpath = "/home/nvidia/Desktop/Intern_project/return_ori_img/cut_Test_img_infloc(03)"
path = "/workspace/Test_img"
txtpath = "/workspace/cut_Test_img_infloc(03)"

fin_loc_dc = {}
cutimg_loc_dc = {}


def y_brightness(thimg ,brightness=100, diff=False):
    '''
    diff = True, 亮度偵測會在接觸到第一個符合閥值的px後,繼續往下尋找
    diff = False, 亮度偵測會在接觸到第一個符合閥值的px後,從另外一個邊界尋找

    舉例: y = 0~30, 
    diff= True,30....>28>27>26>......
    diff= False,30....>28>0>1>2>....
    '''
    touch_botten = False
    thimg = cv2.cvtColor(thimg, cv2.COLOR_GRAY2RGB)
    thimg = cv2.cvtColor(thimg, cv2.COLOR_RGB2HSV)

    for val in range(0, thimg.shape[ 0 ]):
        now_y = thimg.shape[0]-val-1
        mean_bright = thimg[now_y, :, 2].mean()
        # print(mean_bright)
        if (mean_bright > brightness) & (touch_botten == False):
            down_px = now_y
            # printtxt = "now_y {} ,mean_bright: {}".format(now_y,mean_bright)
            # print(printtxt)
            touch_botten = True

        if touch_botten == True:
            if mean_bright < brightness:
                if diff is True:
                    up_px=now_y
                    # printtxt = "now_y {} ,mean_bright: {}".format(now_y,mean_bright)
                    # print(printtxt)
                    return down_px,up_px
                if diff == False:
                    for val in range(0, thimg.shape[ 0 ]):
                        # print(val)
                        now_y = val +1
                        mean_bright = thimg[now_y, :, 2].mean()
                        if (mean_bright > brightness) :
                            up_px = now_y
                            return down_px,up_px
    #如果亮度判斷一直沒有抓到上下限值, 則輸出None, 並且後面程式會在是否為空
    down_px = None
    up_px = None
    return down_px , up_px

def x_brightness(thimg ,brightness=100, diff=False):
    '''
    diff = True, 亮度偵測會在接觸到第一個符合閥值的px後,繼續往下尋找
    diff = False, 亮度偵測會在接觸到第一個符合閥值的px後,從另外一個邊界尋找

    舉例: y = 0~30, 
    diff= True,30....>28>27>26>......
    diff= False,30....>28>0>1>2>....
    '''
    touch_botten = False

    thimg = cv2.cvtColor(thimg, cv2.COLOR_GRAY2RGB)
    thimg = cv2.cvtColor(thimg, cv2.COLOR_RGB2HSV)
    for val in range(0, thimg.shape[ 1 ]):
        now_x = thimg.shape[1]-val-1
        mean_bright = thimg[:, now_x, 2].mean()
        if (mean_bright > brightness) & (touch_botten == False):
            right_px = now_x
            # printtxt = "now_x {} : {}".format(now_x,mean_bright)
            # print(printtxt)
            touch_botten = True

        if touch_botten == True:
            if mean_bright < brightness:
                if diff is True:
                    left_px=now_x
                    # printtxt = "now_x {} : {}".format(now_x,mean_bright)
                    # print(printtxt)
                    return left_px,right_px
                if diff == False:
                    for val in range(0, thimg.shape[ 1 ]):
                    # print(val)
                        now_x = val+1
                        mean_bright = thimg[:, now_x, 2].mean()
                        if (mean_bright > brightness) :
                            left_px = now_x
                            return left_px,right_px
    #如果亮度判斷一直沒有抓到上下限值, 則輸出None, 並且後面程式會在是否為空
    left_px = None
    right_px = None
    return left_px , right_px

def get_pcb(img):
    act_img = img.copy()
    h,w,_= act_img.shape

    hsv = cv2.cvtColor(act_img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # 100~150 is the range of green in hue
    lower_blue = np.array([100*0.705, 40, 20])#lower_blue
    upper_blue = np.array([150*0.705, 255, 200])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # mask有很多小白點 嘗試去除,5為測試的數值
    threshold = h/5 * w/5
    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) #计算轮廓所占面积
        if area < threshold:                         #将area小于阈值区域填充黑色
            cv2.drawContours(mask,[contours[i]],-1, (0 , 0 ,0), thickness=-1)     
            continue
    
    if np.all(mask == 0) :
        print("mask return")
        return

    down_px, up_px = y_brightness(mask , 1, diff=False)
    left_px, right_px = x_brightness(mask , 1, diff=False)
    # printtxt="down_px: {} up_px: {} left_px: {} right_px: {} ".format(down_px, up_px, left_px, right_px)
    # print(printtxt)

    # 計算亮度可能因為參數計算, 沒有回傳值, 因此在程式中預先填上None, 並且在後方判斷是否為0
    if (down_px == None) |(up_px == None) | (left_px == None) | (right_px == None) :
        return
    y_extend = h*0.01
    x_extend = w*0.01
    #img_croprange = imgcr
    imgcr=[ up_px - y_extend, 
                    down_px + y_extend,
                    left_px -x_extend,
                    right_px + x_extend]
    # 限制範圍，避免超出圖片原始大小
    # np.clip(imgcr[:1], 0 ,h)
    # np.clip(imgcr[2:], 0 ,w)
    # crop_PCB = act_img[ int(imgcr[0]):int(imgcr[1]),
    #                     int(imgcr[2]):int(imgcr[3]) ] #完整PCB圖片
    crop_PCB = act_img[ int(up_px):int(down_px),
                        int(left_px):int(right_px) ] #完整PCB圖片
    if  np.all(crop_PCB == 0) :
        print("crop_PCB return")
        return
    else :
        cv2.namedWindow('crop_PCB',cv2.WINDOW_NORMAL)
        cv2.imshow('crop_PCB',crop_PCB)
        # cv2.waitKey(0)

        # cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
        # cv2.imshow('mask',mask)
        return crop_PCB,left_px,up_px

def get_finger(crop_PCB):
    act_img = crop_PCB.copy()
    h,w,_= act_img.shape
    hsv = cv2.cvtColor(act_img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([20*0.705, 0, 20])#lower_blue
    upper_blue = np.array([40*0.705, 200, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.namedWindow('fingermask',cv2.WINDOW_NORMAL)
    # cv2.imshow('fingermask',mask)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(act_img,act_img, mask= mask)

    # mask有很多小白點 嘗試去除
    threshold = h/100 * w/100
    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) #计算轮廓所占面积
        if area < threshold:                         #将area小于阈值区域填充黑色
            cv2.drawContours(mask,[contours[i]],-1, (0 , 0 ,0), thickness=-1)     
            continue
    # cv2.namedWindow('fingermask',cv2.WINDOW_NORMAL)
    # cv2.imshow('fingermask',mask)        
    down_px, up_px = y_brightness(mask , 10, diff=False)
    left_px, right_px = x_brightness(mask , 5, diff=False)

    cut_range = [left_px, up_px, right_px, down_px]
    # print(cut_range)
    fin_loc_dc[file_name] = [left_px, up_px, right_px, down_px]

    crop_finger = act_img[up_px:down_px, left_px:right_px] #完整PCB圖片

    # cv2.namedWindow('crop_finger',cv2.WINDOW_NORMAL)
    # cv2.imshow('crop_finger',crop_finger)
    return crop_finger, cut_range

def cut_cropimg(crop_img, file_name, blcbackgroung=blcbackgroung):
    crop_img_shape= crop_img.shape
    for j in range (0, 7):
        x2=np.clip(((j + 2)* 304), 0, crop_img_shape[1])

        cut_img= crop_img[:,( j * 304 ): x2]

        cut_img_h,cut_img_w,_=cut_img.shape
        y_loc = int(304-(cut_img_h/2))

        tar_back= blcbackgroung.copy()
        tar_back[ y_loc : y_loc+cut_img_h, 0:cut_img_w,:]=cut_img #create a black image
  
        outpath = "/home/nvidia/Desktop/Intern_project/return_ori_img/cut_Test_img("+ file_name+ ")"
        outputImg_name=(CONV_OUT_PATH+ file_name+ "_cut"+ str(j)+ ".jpg")
        cv2.imwrite(outputImg_name, tar_back)

        cutimg_loc_dc[file_name+ "_cut"+ str(j)]=[j * 304, 0, y_loc]
        pass

#--------------------------find gold finger and cut img--------------------------
for filename in os.listdir(path):
    file_name = os.path.splitext(filename)[0]#取得檔名

    img_name = (path +"/"+file_name+".jpg")
    img=cv2.imread(img_name)  #read image 

    crop_PCB ,pcb_x, pcb_y= get_pcb(img)
    if crop_PCB.all() != None:
        crop_finger, cut_range= get_finger(crop_PCB)
    cv2.namedWindow("55",cv2.WINDOW_NORMAL)
    cv2.imshow("55",crop_finger)
    cv2.waitKey(0)
    cut_cropimg(crop_finger,file_name,blcbackgroung)

#--------------------------inference--------------------------
# inference = ['',
#              '',
#              '',
#              '{"class":"NG","confidence":94.18,"bbox":[460,280,486,297]}',
#              '{"class":"NG","confidence":97.32,"bbox":[154,280,184,298]}',
#              '',
#              '',   ]
for filename in os.listdir(path):
    file_name = os.path.splitext(filename)[0]#取得檔名

    img_name = (path +"/"+file_name+".jpg")
    img=cv2.imread(img_name)  #read image 
    tar_img = img.copy()

    for idx in range(0, 7):
        txt_name =txtpath+ "/"+ file_name+ "_cut"+ str(idx)+ ".txt"
        with open(txt_name, 'r') as f:
            temp=f.readlines()
            for line in temp:
                cnt= line.strip()#strip()方法用於移除字符串頭尾指定的字符
                if (len(cnt)>0) and cnt !='{"bbox":"Not detection."}':
                    cnt=cnt.split("[")[1]
                    cnt=cnt.split("]")[0]
                    # print(cutimg_loc_dc[file_name+ "_cut"+ str(idx)])

                    infer_loc=[]
                    for loc_num in cnt.split(","):
                        infer_loc.append(loc_num)
                    #print(infer_loc)
                    #change the element in list to int  
                    infer_loc=list(map(int,infer_loc))
                    tar_filename = file_name+ "_cut"+ str(idx)

                    '''
                    cutimg_loc_dc[tar_filename][2] means that original img in order to be trained with yolo
                    therefore we add blackground and change img size to 608*608
                    so we collect every img's movement and save as cutimg_loc_dc[tar_filename][2]
                    '''
                    tarx1 = fin_loc_dc[file_name][0]+ cutimg_loc_dc[tar_filename][0]+ infer_loc[0]+ pcb_x
                    tary1 = fin_loc_dc[file_name][1]- cutimg_loc_dc[tar_filename][2]+ infer_loc[1]+ pcb_y
                    tarx2 = fin_loc_dc[file_name][0]+ cutimg_loc_dc[tar_filename][0]+ infer_loc[2]+ pcb_x
                    tary2 = fin_loc_dc[file_name][1]- cutimg_loc_dc[tar_filename][2]+ infer_loc[3]+ pcb_y
                    cv2.rectangle(tar_img, (tarx1, tary1), (tarx2, tary2), (0, 255, 0), 2)
            # print(tarx1, tary1, tarx2, tary2)

cv2.namedWindow("tar_img",cv2.WINDOW_NORMAL)
cv2.imshow("tar_img",tar_img)
# print(cutimg_loc_dc)
cv2.waitKey(0)
cv2.destroyAllWindows