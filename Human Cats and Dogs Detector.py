import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade=cv2.CascadeClassifier('mydogdetector.xml')
face_cascade2=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade3=cv2.CascadeClassifier('mycatdetector2.xml')

img=cv2.imread('family.bmp')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
font=cv2.FONT_HERSHEY_SIMPLEX
faces=face_cascade.detectMultiScale(gray,1.345,5,75)
faces2=face_cascade2.detectMultiScale(gray,1.3,5)
faces3=face_cascade3.detectMultiScale(gray,1.3,2,75)

for(x,y,w,h) in faces:
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.putText(img,'Dog',(x,y),font,0.9,(0,255,0),2)

for(z,v,b,n) in faces2:
	img=cv2.rectangle(img,(z,v),(z+b,v+n),(0,0,255),2)
	cv2.putText(img,'Human',(z,v),font,0.9,(0,0,255),2)

for(q,w,e,r) in faces3:
	img=cv2.rectangle(img,(q,w),(q+e,w+r),(255,0,0),2)
	cv2.putText(img,'Cat',(q,w),font,0.9,(255,0,0),2)
	
p,l,m=cv2.split(img)
img=cv2.merge([m,l,p])

plt.imshow(img)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

