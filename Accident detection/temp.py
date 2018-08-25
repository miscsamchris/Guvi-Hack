#Import necessary packages
import cv2
import math, operator
import functools
import smtplib
def sendmail():
    sender = 'infantsamchris@gmail.com'
    receivers = ['balavignesh25@gmail.com']
    message = """From: From Person <from@fromdomain.com>
    To: To Person <to@todomain.com>
    MIME-Version: 1.0
    Content-type: text/html
    Subject: SMTP HTML e-mail test
    
    This is an e-mail message to be sent in HTML format
    
    <b>This is HTML message.</b>
    <h1>This is headline.</h1>
    """
    
    try:
       smtpObj = smtplib.SMTP('localhost')
       smtpObj.sendmail(sender, receivers, message)         
       print ("Successfully sent email")
    except Exception:
       print ("Error: unable to send email")

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
  if (rms<3250):
       print("Accident")
  #Updates the frames
  t_minus = t
  t = t_plus
  t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
  
  #Destroys the window after key press
  key = cv2.waitKey(10)
  if key == 27:
    cv2.destroyWindow(winName)
    cv2.destroyWindow("Video")
    break  