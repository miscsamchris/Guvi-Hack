# -*- coding: utf-8 -*-

import cv2
print(cv2.__version__)

cascade_src = 'cars.xml'
bus_src='Bus_front.xml'
motorbike='two_wheeler.xml'

video_src = 'C:/Users/Sam Christian/Desktop/Guvi Hack/road.mp4'
    
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
bus_cascade = cv2.CascadeClassifier(bus_src)
mb_cascade = cv2.CascadeClassifier(motorbike)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    buses = car_cascade.detectMultiScale(gray, 1.1, 1)
    bikes = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)  
    for (x,y,w,h) in buses:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)   
    for (x,y,w,h) in bikes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   
        
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break
cap.release()
cv2.destroyAllWindows()