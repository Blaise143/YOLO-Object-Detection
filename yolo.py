import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


# Loading Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    result = model(frame)
    cv2.imshow('YOLO', np.squeeze(result.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()