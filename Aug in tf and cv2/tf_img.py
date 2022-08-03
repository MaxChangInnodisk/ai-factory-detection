import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import math

#----------test infromation ----------
CONV_OUT_PATH='/home/nvidia/Desktop/opencvTest/changebright/'

path = "/home/nvidia/Desktop/findfinger/NG/ng_defect_cutby6"
file_name="12_cut1.jpg"
#----------test infromation ----------

BGRimg=cv.imread("/home/nvidia/Desktop/opencvTest/messi.jpg")
RGB_img=cv.cvtColor(BGRimg,cv.COLOR_BGR2RGB)

# image_raw_data = tf.io.gfile.GFile("/home/nvidia/Desktop/opencvTest/messi.jpg", 'rb').read()
# img=tf.image.decode_jpeg(image_raw_data)

with tf.compat.v1.Session() as sess:
    image_raw_data = tf.io.gfile.GFile("/home/nvidia/Desktop/opencvTest/messi.jpg", 'rb').read()
    image_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.flip_up_down(image_data)
    plt.imshow(image_data.eval())
    plt.show()

    # t1= time.time()
    # for i in range(0, 8):

    #     # img_data = tf.image.flip_up_down(RGB_img)
    #     # img_data = tf.image.random_contrast(RGB_img,lower=0.8,upper=3)
    #     # img_data = tf.image.flip_left_right(RGB_img)
    #     # img_data = tf.image.random_brightness(RGB_img,max_delta=0.5)
    #     # img_data = tf.image.random_saturation(RGB_img,lower=0,upper=4)
    #     img_data = tf.contrib.image.rotate(RGB_img,lower=0,upper=4)

    # t2= time.time()
    # print("TF耗时",t2-t1)
    # plt.imshow(img_data.eval())
    # plt.show()

def yolotoxy(img,path):
    h,w,_=img.shape
    with open(path+ "/"+ file_name.split(".")[0] +".txt", 'r') as f:
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
    # print(bboxes)
    return bboxes


def brightness(img,diff=True,v_up=155,v_low=100,step=10):

    '''
    Major design:
    沒有BBoxes的情況:BBoxes有無預設值
    (在程式中設計時,有給filename時自動偵測BBoxes並計算BBoxes附近的平均亮度,
    如果沒給就忽略亮度黑暗的區域,且做亮度平均計算)

    Other:
    The imege read bycv2 is defauly as BGR,
    need to change HSV ,because HSV's third channel is brightness
    
    # 按下任意鍵則關閉所有視窗
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    reference:https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    '''

    assert v_up > v_low
    assert v_up < 255 and v_low > 0

    h,w,_=img.shape
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    ori_hsv_h, ori_hsv_s, ori_hsv_v = cv.split(hsv)

    if diff==True :
        #initial info
        txt_name=(path+ "/"+ file_name.split(".")[0]+ ".txt")
        with open(txt_name, 'r') as f:
            temp=f.readlines()
            line=temp[0] #get fist bboxes
            line=line.strip()
            line=line.split(" ")
            #Save the paremeter in the Yolo text
            x_,y_,w_,h_=eval(line[1]),eval(line[2]),eval(line[3]),eval(line[4]) 

            x1=w*x_- 0.5* w* w_
            x2=w*x_+ 0.5* w* w_
            y1=h*y_- 0.5* h* h_
            y2=h*y_+ 0.5* h* h_
            bboxes=[x1, y1, x2, y2, 0]

            #let bboxes have limit
            if x1-20<0:
                x1=20
            if x2+20>h:
                x2=h-20
            if y1-20<0:
                y1=20
            if y2-20>w:
                y2=w-20        

            x1=int(x1)
            x2=int(x2)
            y1=int(y1)
            y2=int(y2)
        
        bbox_hsv = cv.cvtColor(img[y1- step: y2+ step, x1- step: x2+ step], cv.COLOR_BGR2HSV)
        
        tar_v=ori_hsv_v
        countnum=round(bbox_hsv[..., 2].mean(), 2)
        # print(countnum)
        while(countnum< v_up):

            countnum+= step
            tar_v = cv.add(tar_v, step)    #image add brightness

            #change to BGR
            final_hsv = cv.merge((ori_hsv_h, ori_hsv_s, tar_v))
            img2 = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
            cv.imshow(str(countnum), img2)
            cv.waitKey(0)
            cv.destroyAllWindows()
            

        tar_v=ori_hsv_v
        countnum=round(bbox_hsv[...,2].mean(), 2)
        while(countnum> v_low):

            countnum-= step
            tar_v = cv.add(tar_v,-step)    #image add brightness

            #change to BGR
            final_hsv = cv.merge((ori_hsv_h, ori_hsv_s, tar_v))
            img2 = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
            cv.imshow(str(countnum), img2)
            cv.waitKey(0)
            cv.destroyAllWindows()

    if diff==False :

        INCREASE=20
        DECREASE=20

        mean_v=np.array(ori_hsv_v)
        mean_v=mean_v[mean_v>0]    #get item> 0 in ori_hsv_v
        A=round(mean_v.mean() , 2)   #get major v in Image

        tar_v=ori_hsv_v
        
        while(A< v_up):
            A+= 20
            tar_v = np.where(tar_v <= 255 - INCREASE, tar_v + INCREASE, 255)

            #change to BGR
            final_hsv = cv.merge((ori_hsv_h, ori_hsv_s, tar_v)) 
            img2 = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
            # cv.imshow(str( A), img2)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

        A=round(mean_v.mean() , 2)
        tar_v=ori_hsv_v
        while(A> v_low):
            A-= 20
            tar_v = np.where(tar_v > DECREASE, tar_v - INCREASE, 0)

            #change to BGR
            final_hsv = cv.merge((ori_hsv_h, ori_hsv_s, tar_v)) 
            img2 = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
            # cv.imshow(str( A), img2)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

def flip_left_right(img,path):
    bboxes=yolotoxy(img,path)
    img_center = np.array(img.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    img = img[:, ::-1, :]
    bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])
    #加上兩倍（中間減掉現在座標），等於鏡像複製到對面

    box_w = abs(bboxes[:, 0] - bboxes[:, 2])

    bboxes[:, 0] -= box_w
    bboxes[:, 2] += box_w
    #此時因為鏡像複製,所以原本的x2=x1,原本的x1=x2,因此加減現在的BOX寬度，求回原本的座標
    return img, bboxes

def flip_up_down(img,path):
    bboxes=yolotoxy(img,path)
    img_center = np.array(img.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    img = img[::-1, :, :]
    bboxes[:, [1, 3]] += 2*(img_center[[1, 3]] - bboxes[:, [1, 3]])
    #加上兩倍（中間減掉現在座標），等於鏡像複製到對面

    box_w = abs(bboxes[:, 1] - bboxes[:, 3])

    bboxes[:, 1] -= box_w
    bboxes[:, 3] += box_w
    #此時因為鏡像複製,所以原本的x2=x1,原本的x1=x2,因此加減現在的BOX寬度，求回原本的座標
    return img, bboxes

def contrast(img, c_low= -80 , c_up= 80 ,step= 20 ):
    for val in range(c_low, c_up+ step, step):
        # print(val)
        brightness = 0
        output = img * (val/127 + 1) - val + brightness # 轉換公式

        output = np.clip(output, 0, 255)
        output = np.uint8(output)

        # cv.imshow('555', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # cv.imshow(str( val ), output)
        # cv.waitKey(0)
        # cv.destroyAllWindows()


# brightness(img,diff=True)
def Saturation(img, s_low= 0 , s_up= 255 ,step= 20):

    assert s_low >= 0 and s_up <= 255
    assert step > 0

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    ori_hsv_h, ori_hsv_s, ori_hsv_v = cv.split(hsv)
    # print( ori_hsv_s.mean())

    # tar_s = ori_hsv_s + 50
    # final_hsv = cv.merge((ori_hsv_h, tar_s, ori_hsv_v)) 
    # img2 = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)


    mean_s = np.array(ori_hsv_s)
    mean_s = mean_s[mean_s>0]    #get item> 0 in ori_hsv_s

    A = round(mean_s.mean() , 2)   #get major v in Image
    tar_s = ori_hsv_s
    
    while(A< s_up):
        A += step
        tar_s = np.where(tar_s <= 255 - step, tar_s + step, 255)

        #change to BGR
        final_hsv = cv.merge((ori_hsv_h, tar_s, ori_hsv_v)) 
        img2 = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        # cv.imshow(str( A), img2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # outputImg_name=(CONV_OUT_PATH+ file_name+ "_changeBright"+str( A )+".jpg")
        # cv.imwrite(outputImg_name, img2)

    A=round(mean_s.mean() , 2)
    tar_s = ori_hsv_s

        

    while(A> s_low):
        A -= step
        tar_s = np.where(tar_s > step, tar_s - step, 0)

        #change to BGR
        final_hsv = cv.merge((ori_hsv_h, tar_s, ori_hsv_v)) 
        img2 = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        
        # cv.imshow(str( A), img2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # outputImg_name=(CONV_OUT_PATH+ file_name+ "_changeBright"+str( A )+".jpg")
        # cv.imwrite(outputImg_name, img2)



# t1= time.time()
# # img_, BBox_= flip_left_right(BGRimg, path)
# # img_, BBox_= flip_up_down(BGRimg, path)   
# # brightness(BGRimg,diff=False)
# # contrast(BGRimg)
# Saturation(BGRimg,70, 180)
# t2= time.time()
# print("cv耗时",t2-t1)

# cv.imshow("555",img_)
# cv.waitKey(0)
# cv.destroyAllWindows 