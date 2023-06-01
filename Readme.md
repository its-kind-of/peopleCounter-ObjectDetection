# Object Detection and Tracking using YOLO and SORT

This project implements object detection and tracking using the YOLO (You Only Look Once) algorithm for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for object tracking. It can be used for various computer vision applications such as counting objects, monitoring crowd movement, and more.

## Installation

1. Clone the repository:

git clone https://github.com/your-username/your-repository.git

1. Install the required dependencies:

```pip install -r requirements.txt```


1. Download the YOLO weights file and place it in the project directory:

yolov8n.pt

4. Download the necessary image masks and graphics files and place them in the project directory:

images/people-masks.png
images/people-graphics.png

5. Run the main script:

python main.py

```## Usage

1. Select the video source:

- To use the webcam, uncomment the following lines in the code:

  ```python
  # cap = cv2.VideoCapture(0)
  # cap.set(3, 1280) # width
  # cap.set(4, 720) # height
  ```

- To use a video file, provide the file path in the following line:

  ```python
  cap = cv2.VideoCapture("path/to/video/file")
  ```

2. Set the desired output width and height (optional):

```python
output_width = 520
output_height = 640```

## Usage

1. Select the video source:

- To use the webcam, uncomment the following lines in the code:

  ```python
  # cap = cv2.VideoCapture(0)
  # cap.set(3, 1280) # width
  # cap.set(4, 720) # height
  ```

- To use a video file, provide the file path in the following line:

  ```python
  cap = cv2.VideoCapture("path/to/video/file")
  ```

2. Set the desired output width and height (optional):

```python
output_width = 520
output_height = 640
Customize the detection and tracking parameters (optional):

python
Copy code
confidence_threshold = 0.3
iou_threshold = 0.3
Define the line limits for counting objects (optional):

```
limitsUp = [120, 161, 330, 140]
limitsDown = [735, 489, 950, 455]```
Run the script and observe the object detection and tracking results.

# Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

# License
This project is licensed under the MIT License.

# Acknowledgements
* YOLO: https://github.com/ultralytics/yolov5
* SORT: https://github.com/abewley/sort
* CVZone: https://github.com/cvzone/cvzone