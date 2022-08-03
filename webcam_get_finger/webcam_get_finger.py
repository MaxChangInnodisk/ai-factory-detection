# import shutil
import cv2 as cv
from cv2 import cvtColor
from cv2 import waitKey
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
import time

IMG_SOURCE_PATH='/home/nvidia/Desktop/findfinger/NG'
CONV_OUT_PATH='/home/nvidia/Desktop/findfinger/Good/'
DATAAUG_TIMES=1
H=608
W=608

file_name='07'
img_name='/home/nvidia/Desktop/Intern_project/return_ori_img/Test_img/01.jpg'
# img_name='/home/nvidia/Desktop/findfinger/Good/webcamtest2.jpg'
img = cv.imread(img_name)
# img = np.rot90(img)

cv.namedWindow('ori',cv.WINDOW_NORMAL)
cv.imshow('ori',img)

blcbackgroung = np.zeros((H,W,3), np.uint8)

def do_GaussianBlur(gray):
    # 高斯模糊(高斯平滑)，與計算核心大小
    kernel_size = 5
    blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    return blur_gray

def do_Canny(blur_gray):
    # Canny邊緣運算
    low_threshold = 15
    high_threshold = 100
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)
    return edges

def y_brightness(thimg ,brightness=100, diff=False):
    '''
    diff = True, 亮度偵測會在接觸到第一個符合閥值的px後,繼續往下尋找
    diff = False, 亮度偵測會在接觸到第一個符合閥值的px後,從另外一個邊界尋找

    舉例: y = 0~30, 
    diff= True,30....>28>27>26>......
    diff= False,30....>28>0>1>2>....
    '''
    touch_botten = False
    thimg = cv.cvtColor(thimg, cv.COLOR_GRAY2RGB)
    thimg = cv.cvtColor(thimg, cv.COLOR_RGB2HSV)

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

    thimg = cv.cvtColor(thimg, cv.COLOR_GRAY2RGB)
    thimg = cv.cvtColor(thimg, cv.COLOR_RGB2HSV)
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

def cut_cropimg(crop_img, file_name,blcbackgroung=blcbackgroung):
    for j in range (0, 7):
        
        cut_img= crop_img[:,( j * 304 ):((j + 2)* 304)]
        
        cut_img_h,_,_=cut_img.shape
        # print(cut_img_h)
        # center = 304-(cut_img_h/2)

        # blcbackgroung[ 200 : 200+cut_img_h,:,:]=cut_img #create a black image

        # outputImg_name=(CONV_OUT_PATH+ file_name+ "_cut"+ str(j)+ ".jpg")
        # cv.imwrite(outputImg_name, cut_img)
        pass


def drawbbox(img, x1, y1, x2, y2,labelname):
    class_name = str(labelname)
    # b_box 左上角坐标
    ptLeftTop = np.array([x1, y1])
    # 文本框左上角坐标
    textleftop = []
    # b_box 右下角坐标
    ptRightBottom =np.array([x2, y2])
    # 框的颜色
    point_color = (0, 255, 0)
    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    # (500, 375, 3) -> h w c
    src = img.copy()
    cv.namedWindow('src', cv.WINDOW_NORMAL)
    src = np.array(src)
    # 画 b_box
    cv.rectangle(src, tuple(ptLeftTop), tuple(ptRightBottom), point_color, thickness, lineType)

    # 获取文字区域框大小
    t_size = cv.getTextSize(class_name, 1, cv.FONT_HERSHEY_PLAIN, 1)[0]
    # 获取 文字区域右下角坐标
    textlbottom = ptLeftTop + np.array(list(t_size))
    # 绘制文字区域矩形框
    cv.rectangle(src, tuple(ptLeftTop), tuple(textlbottom),  point_color, -1)
    # 计算文字起始位置偏移
    ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)
    # 绘字
    cv.putText(src, class_name , tuple(ptLeftTop), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    # 打印图片的shape
    # print(src.shape)

    cv.imshow('src', src)

def get_pcb(img):
    act_img = img.copy()
    h,w,_= act_img.shape

    hsv = cv.cvtColor(act_img, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # 100~150 is the range of green in hue
    lower_blue = np.array([100*0.705, 40, 20])#lower_blue
    upper_blue = np.array([150*0.705, 255, 200])

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # mask有很多小白點 嘗試去除,5為測試的數值
    threshold = h/5 * w/5
    contours,_=cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i]) #计算轮廓所占面积
        if area < threshold:                         #将area小于阈值区域填充黑色
            cv.drawContours(mask,[contours[i]],-1, (0 , 0 ,0), thickness=-1)     
            continue
    
    if np.all(mask == 0) :
        print("mask return")
        return

    down_px, up_px = y_brightness(mask , 1, diff=False)
    left_px, right_px = x_brightness(mask , 1, diff=False)
    printtxt="down_px: {} up_px: {} left_px: {} right_px: {} ".format(down_px, up_px, left_px, right_px)
    print(printtxt)

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
        cv.namedWindow('crop_PCB',cv.WINDOW_NORMAL)
        cv.imshow('crop_PCB',crop_PCB)
        # cv.waitKey(0)

        # cv.namedWindow('mask',cv.WINDOW_NORMAL)
        # cv.imshow('mask',mask)
        return crop_PCB

def get_ic(crop_PCB):
    act_img = crop_PCB.copy()
    h,w,_= act_img.shape
    blcbackgroung = np.zeros((h,w,3), np.uint8)

    # _, mask= get_mask(act_img, 150 ,180)
    hsv = cv.cvtColor(act_img, cv.COLOR_BGR2HSV)
    lower_blue = np.array([130*0.705, 0, 0])#lower_blue
    upper_blue = np.array([180*0.705, 100, 100])

    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # cv.namedWindow('fingermask',cv.WINDOW_NORMAL)
    # cv.imshow('fingermask',mask) 
    # Bitwise-AND mask and original image
    # res = cv.bitwise_and(act_img,act_img, mask= mask)
    # mask有很多小白點 嘗試去除,10為測試的數值 可以在根據IC的大小往下調整
    threshold = h/10 * w/10
    contours,_=cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i]) #计算轮廓所占面积
        if area < threshold:                         #将area小于阈值区域填充黑色
            cv.drawContours(mask,[contours[i]],-1, (0 , 0 ,0), thickness=-1)     
            continue
    # cv.namedWindow('fingermask',cv.WINDOW_NORMAL)
    # cv.imshow('fingermask',mask)        
    contours,_ = cv.findContours(mask.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt=[] #儲存面積大小的陣列
    for i in range(len(contours)):
        area = cv.contourArea(contours[i]) #计算轮廓所占面积
        cnt.append(area)
    sort_cnt= sorted(cnt, reverse=True) #將取出的面積大小排序
    if len(sort_cnt) ==0:
        return 
    tar_area = sort_cnt[0]*0.9 #目標IC的面積大小

    ic_number=0
    for i in range(len(contours)):
        area = cv.contourArea(contours[i]) #计算轮廓所占面积
        if area > tar_area :
            # print(area)
            ic_number +=1
            #將IC區域標記白色
            draw_label = cv.drawContours(blcbackgroung.copy(),contours,i,(255,255,255),-1)
            #轉成灰階計算二值化
            draw_label_gray = cv.cvtColor(draw_label.copy(), cv.COLOR_BGR2GRAY)
            _, label_0ths = cv.threshold(draw_label_gray, 254, 0, cv.THRESH_TOZERO)


            # 0.01跟0.02是根據此次PCB設定的亮度數值
            # tarb : 目標亮度(跟輸入的IC大小浮動)
            label_0ths_shape=label_0ths.shape
            y_tarb,x_tarb = label_0ths_shape[0]*0.01 ,label_0ths_shape[1]*0.01
         
            down_px, up_px = y_brightness(label_0ths , y_tarb, diff=False)
            left_px, right_px = x_brightness(label_0ths , x_tarb, diff=False)
            # down_px, up_px = y_brightness(label_0ths , 10)
            # left_px, right_px = x_brightness(label_0ths , 10)
            # print(str(area),up_px, down_px ,left_px ,right_px)
            ic_img = crop_PCB[up_px: down_px, left_px : right_px ] #完整IC圖片

            # cv.namedWindow(str(ic_number),cv.WINDOW_NORMAL)
            # cv.imshow(str(ic_number), ic_img)

def get_finger(crop_PCB):
    act_img = crop_PCB.copy()
    h,w,_= act_img.shape
    hsv = cv.cvtColor(act_img, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([20*0.705, 0, 20])#lower_blue
    upper_blue = np.array([40*0.705, 200, 255])
    
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    cv.namedWindow('fingermask',cv.WINDOW_NORMAL)
    cv.imshow('fingermask',mask)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(act_img,act_img, mask= mask)

    # mask有很多小白點 嘗試去除
    threshold = h/100 * w/150
    contours,_=cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv.contourArea(contours[i]) #计算轮廓所占面积
        if area < threshold:                         #将area小于阈值区域填充黑色
            cv.drawContours(mask,[contours[i]],-1, (0 , 0 ,0), thickness=-1)     
            continue
    cv.namedWindow('mask',cv.WINDOW_NORMAL)
    cv.imshow('mask',mask)
    down_px, up_px = y_brightness(mask , 5, diff=False)
    left_px, right_px = x_brightness(mask , 5, diff=False)
    # print(left_px, right_px)

    crop_finger = act_img[up_px:down_px, left_px:right_px] #完整PCB圖片

    cv.namedWindow('crop_finger',cv.WINDOW_NORMAL)
    cv.imshow('crop_finger',crop_finger)

## ------靜態測試------
crop_PCB = get_pcb(img)
if crop_PCB.all() != None:
    get_ic(crop_PCB)
    get_finger(crop_PCB)
cv.waitKey(0)
cv.destroyAllWindows

## ------動態測試------
# cap = cv.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     crop_PCB = get_pcb(frame)
#     if crop_PCB is not None:
#         get_ic(crop_PCB)
#         get_finger(crop_PCB)

#     cv.imshow('frame', frame)
#     # time.sleep(0.1)

#     key = cv.waitKey(1)
#     if key == 27:
#         break
# cap.release()
# cv.destroyAllWindows()

