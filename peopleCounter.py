from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import numpy as np

# WebCam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280) # width
# cap.set(4, 720) # height

# Video
cap = cv2.VideoCapture(r"D:\open-cv\yolo-projects\videos\people-on-escalator-1.mp4")

model = YOLO(r"D:\open-cv\yolo-projects\Yolo-Weights\yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hyadrant",
              "stop sign ", "parking meter", "bench", "bird", "cat", "dog", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
              "skateboard", "surfboard", "tennis racket",  "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dinning table", "toilet", "tv monitor",
              "laptop", "mouse", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
              ]

# Specify the desired output width and height (1280 X 720)
output_width = 520
output_height = 640
imgMask = cv2.imread(r"D:\open-cv\yolo-projects\images\people-masks.png")

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [120, 161, 330, 140]
limitsDown = [735, 489, 950, 455]

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()

    # # Crop the image
    # imgMask = imgMask[0:img.shape[0], 0:img.shape[1]]

    imgRegion = cv2.bitwise_and(img, imgMask)

    imgGraphics = cv2.imread(r"D:\open-cv\yolo-projects\images\people-graphics.png", cv2.IMREAD_UNCHANGED)

    # Convert imgGraphics to BGRA
    imgGraphicsBGRA = cv2.cvtColor(imgGraphics, cv2.COLOR_BGR2BGRA)

    imgRegion = cvzone.overlayPNG(imgRegion, imgGraphicsBGRA, (1050, 0))

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    totalCount = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # bound box for cv
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # bound box for cvzone
            # x1, y1, w, h = box.xywh[0]
            w, h = x2-x1, y2-y1
            # bbox = int(x1), int(y1), int(w), int(h)

            # confidence
            confidence = math.ceil((box.conf[0] * 100))/100

            # class name
            className = int(box.cls[0])
            currentClass = classNames[className]

            allowedObjects = ["person"]

            if currentClass in allowedObjects and confidence > 0.3:
                # cvzone.cornerRect(imgRegion, (x1, y1, w, h), l=9, rt=5)
                # cvzone.putTextRect(imgRegion, f"{confidence, currentClass}", (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=3)
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(imgRegion, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(imgRegion, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(imgRegion, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(imgRegion, f"{int(Id)}", (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(imgRegion, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 30 < cy < limitsUp[1] + 30:
            if totalCountUp.count(Id) == 0:
                totalCountUp.append(Id)
                cv2.line(imgRegion, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 30 < cy < limitsDown[1] + 30:
            if totalCountDown.count(Id) == 0:
                totalCountDown.append(Id)
                cv2.line(imgRegion, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(imgRegion, f"Count : {len(totalCount)}", (50, 50), scale=2, thickness=3, offset=10)
    cv2.putText(imgRegion, f"{str(len(totalCountUp))}", (1150, 315), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 7)
    cv2.putText(imgRegion, f"{str(len(totalCountDown))}", (1415, 315), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 7)


    # Resize the frame
    # resized_frame = cv2.resize(img, (output_width, output_height))
    # cv2.imshow("Image", img)
    cv2.imshow("Image-Region", imgRegion)

    cv2.waitKey(1)
