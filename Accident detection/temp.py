#Import necessary packages
import cv2
import math, operator
import functools
def sendmail():
        print("Accident: Send message to Control Room")

#Function to find difference in frames
def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

#Import video from webcam
cam = cv2.VideoCapture("C:/Users/Sam Christian/Desktop/Guvi Hack/rem.mkv")

#Creating window to display 
winName = "Accident Detector"
cv2.namedWindow(winName)
cv2.namedWindow("Video")
#Reading frames at multiple instances from webcam to different variables
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
cv2.imwrite("C:/Users/Sam Christian/Desktop/Guvi Hack/shotp.jpg",t)
cascade_src = 'cars.xml'
bus_src='Bus_front.xml'
motorbike='two_wheeler.xml'

video_src = 'C:/Users/Sam Christian/Desktop/Guvi Hack/road.mp4'
    
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
bus_cascade = cv2.CascadeClassifier(bus_src)
mb_cascade = cv2.CascadeClassifier(motorbike)
while True:
  #Display video out through the window we created
  cv2.imshow( winName, diffImg(t_minus, t, t_plus) )
  cv2.imshow("Video",t)
  #Calling function diffImg() and assign the return value to 'p'
  p=diffImg(t_minus, t, t_plus)
  
  #Writing 'p' to a directory
  cv2.imwrite("C:/Users/Sam Christian/Desktop/Guvi Hack/shot.jpg",p)
  
   #From Python Image Library(PIL) import Image class
  from PIL import Image
   
   #Open image from the directories and returns it's histogram's
  h1 = Image.open("C:/Users/Sam Christian/Desktop/Guvi Hack/shotp.jpg").histogram()
  h2 = Image.open("C:/Users/Sam Christian/Desktop/Guvi Hack/shot.jpg").histogram()
 
   #Finding rms value of the two images opened before		
  rms = math.sqrt(functools.reduce(operator.add,map(lambda a,b: (a-b)**2, h1, h2))/len(h1))  
   #If the RMS value of the images are under our limit 
  print(rms)
  if (rms<3160):
       print("Accident")
       sendmail()
  #Updates the frames
  t_minus = t
  t = t_plus
  t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
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
  #Destroys the window after key press
  key = cv2.waitKey(10)
  if key == 27:
#    cv2.destroyWindow(winName)
#    cv2.destroyWindow("Video")
      cv2.destroyAllWindows()
    break  