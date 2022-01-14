# Photo (Photo Recognition)

import cv2
from random import randrange

imgFile = r'Tony.jpg'
cascPath = "haarcascade_eye.xml"

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

# Choose an image to detect face
img = cv2.imread(imgFile)

# Convert to Grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinator = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinator:
  cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2 ) # ( , Start, Length, Color, Border )

#
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey(0)

print("Code Complete")