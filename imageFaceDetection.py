import cv2
from blazeFaceDetector import blazeFaceDetector

scoreThreshold = 0.7
iouThreshold = 0.3

imagePath = "image.jpg"

# Initialize face detector
faceDetector = blazeFaceDetector("front", scoreThreshold, iouThreshold)

# Read RGB images
img = cv2.imread(imagePath, cv2.IMREAD_COLOR)

# Detect faces
detectionResults = faceDetector.detectFaces(img)

# Draw detections
img_plot = faceDetector.drawDetections(img, detectionResults)
cv2.imshow("detections", img_plot)
cv2.waitKey(0)
