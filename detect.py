from ultralytics import YOLO
import cv2
import cvzone #display all detections
import math
from playsound import playsound

cap = cv2.VideoCapture(0)
cap.set(3, 720) #width
cap.set(4, 720) #height

prev_cls = -1
model = YOLO("../Yolo-weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike","aeroplane","bus","train","truck","boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
              "cat", "dog","horse","sheep","cow", "elephant", "bear","zebra","giraffe","backpack",
              "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","spoerts ball",
              "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
              "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
              "sandwich","orange","brocolli","carrot","hot dog","pizza","donut","cake","chair",
              "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
              "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
              "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
            ]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        if not len(list(boxes)):
            print("No Detection")
            prev_cls = -1
        print("Box Found")

        #bounding box creation
        for box in boxes:
            #for cv
            #x1,y1, x2, y2= box.xyxy[0]
            # x1, y1, x2, y2= int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            #cv2.rectangle(img,(x1,y1), (x2, y2),(255,0,255),3)
            #cvzone.cornerRect(img, bbox)

            #for cvzone
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            cvzone.cornerRect(img,(x1,y1,w,h)) #cant access cornerRect

            #confidence values
            conf = math.ceil((box.conf[0]*100))/100
            # print(conf)

            #class name
            cls = int(box.cls[0])
            # print(cls)

            if(conf>0.6 and cls!=prev_cls):
                print("Previous Class : ",prev_cls)
                print("Class Name : ",classNames[cls])
                playsound('audio/'+classNames[cls]+'.mp3')
                prev_cls = cls

            # creating a box for displaying confidence value and class name
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale = 1, thickness = 1)

    cv2.imshow("Image", img)
    cv2.waitKey(100)

