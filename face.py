import cv2

#Trained data set
trainedDataset= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Read a Image
img=cv2.imread('images/stanley.jpg')

#Convert into grayscale
gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
faces=trainedDataset.detectMultiScale(gray)
print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('Stanley',img)


#cv2.imshow('Gray',gray)
cv2.waitKey()


