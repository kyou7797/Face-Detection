# Video (Photo Recognition)

import cv2
from random import randrange

cascPath = "haarcascade_eye.xml"
smilePath = "haarcascade_smile.xml"
facePath = "haarcascade_frontalface_default.xml"

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_eye_data = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + facePath)
trained_smile_data = cv2.CascadeClassifier(cv2.data.haarcascades + smilePath)

# Choose an image to detect face
webcam = cv2.VideoCapture(0) # in (Link Video)

while True:
  succesful_frame_read, frame = webcam.read()

  # Convert to Grayscale
  grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect Face
  face_coordinator = trained_face_data.detectMultiScale(grayscaled_img)

  # for (x1, y1, w1, h1) in eye_coordinator:
  #   cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2 ) # ( , Start, Length, Color, Border )

  for (x2, y2, w2, h2) in face_coordinator:
    cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)

    faceArea = frame[y2:y2+h2, x2:x2+w2]

    FaceGray = cv2.cvtColor(faceArea, cv2.COLOR_BGR2GRAY)

    # Detect Smile
    smile_coordinator = trained_smile_data.detectMultiScale(FaceGray, scaleFactor = 1.7, minNeighbors = 20)

    # Detect Eye
    eye_coordinator = trained_eye_data.detectMultiScale(FaceGray, scaleFactor = 1.3, minNeighbors = 10)

    # Draw Rectangle Smile
    for (x, y, w, h) in smile_coordinator:
      cv2.rectangle(faceArea, (x, y), (x+w, y+h), (0, 0, 255), 2 )
    
    for (x, y, w, h) in eye_coordinator:
      cv2.rectangle(faceArea, (x, y), (x+w, y+h), (255, 255, 255), 2 )

    if len(smile_coordinator) > 0:
      cv2.putText(frame, 'smiling', (x2, y2 + h2 + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

  cv2.imshow('Clever Programmer Face Detector', frame)
  key = cv2.waitKey(1)

  # Stop if Press Q key
  if key == 81 or key == 113:
    break

webcam.release()
