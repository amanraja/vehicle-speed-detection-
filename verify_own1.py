import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
cap = cv2.VideoCapture(r'D:\projects\safe\video\cutvideo.mp4')
font=cv2.FONT_HERSHEY_SIMPLEX
ret, frame = cap.read()
frame=frame[:,200:]
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
polygon1 = Polygon([[42,316],[173,297],[210,348],[63,361]])
kernel1 = np.array([[1,1],[1,1]])
fcount=0
a=[]
r=0
speed=0
count=0
c=0
#cap.get(3),cap.get(4))
OverSpeed=[]
fourc=cv2.VideoWriter_fourcc(*'XVID')
out1=cv2.VideoWriter(r'D:\projects\safe\video\speed.mp4',fourc,30.0,(640,480))
out2=cv2.VideoWriter(r'D:\projects\safe\video\shadow.mp4',fourc,30.0,(640,480))
out3=cv2.VideoWriter(r'D:\projects\safe\video\wshadow.mp4',fourc,30.0,(640,480))
def shadow(frame):
    gimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(gimg,140,255,cv2.THRESH_BINARY)
    fg=cv2.bitwise_and(th,th,mask=fgmask)
    return fg
while(ret):
    fcount=fcount+1
    ret, frame = cap.read()
    frame=frame[:,200:]
    img=frame.copy()
    pts1=np.array([[42,316],[173,297],[210,348],[63,361]],np.int32)
    pts1=pts1.reshape((-1,1,2))
    cv2.polylines(frame,[pts1],1,(0,255,255))
    fgmask = fgbg.apply(frame)
    shad=shadow(frame)
    cv2.imshow('window1',fgmask)
    cv2.imshow('window2',shad)
    out2.write(fgmask)
    out3.write(shad)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel1)
    fgmask=cv2.erode(fgmask,kernel1,iterations=2)
    fgmask=cv2.dilate(fgmask, kernel1, iterations=10)
    gimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(gimg,140,255,cv2.THRESH_BINARY)
    fg=cv2.bitwise_and(th,th,mask=fgmask)
    img[:,:,2]=cv2.add(img[:,:,2],fg)
    #cv2.imshow('red',img)
    #cv2.imshow('window2',fg)
    ## speed and contour ##
    (_,cnts,_)=cv2.findContours(fg.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        (x, y, w, h)=cv2.boundingRect(contour)
        if w>=30 and h>=30:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 1)
            cv2.circle(frame,(int(x+w/2),int(y+(h)/2)),5,(0,0,255),-1)
            if(polygon1.contains(Point(int(x+w/2),int(y+(h)/2)))):
                #print('yes')
                a.append(fcount)
            else:
                pass
                #print('NO')
    dist=4.6
    fps=30
    if len(a)>1:
        if fcount-max(a)>1:
            #print(a)
            speed=((dist*fps)/(max(a)-min(a)))*(18/5)
            count=count+1
            print(a,speed)
            a=[]
    s="SPEED="+str(round(speed,2))
    cv2.putText(frame,s,(0,100),font,1,(0,255,0),2,cv2.LINE_AA)
    out1.write(frame)
    cv2.imshow('window3',frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()
out1.release()
out2.release()
out3.release()
cv2.destroyAllWindows()
