import cv2

cap = cv2.VideoCapture(0)
print("Camera functional:", cap.isOpened())
cap.release()
