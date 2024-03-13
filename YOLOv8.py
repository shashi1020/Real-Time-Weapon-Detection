import numpy as np
import cv2
from ultralytics import YOLO
import random

def predictions(img, Detect_obj, Box_colours, class_list):
    DP = Detect_obj[0].numpy()

    if len(DP) != 0:
        for i in range(len(Detect_obj[0])):
            boxes = Detect_obj[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                img,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                Box_colours[int(clsID)],
                3,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img,
                class_list[int(clsID)]
                + " "
                + str(round(conf, 3))
                + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 0, 255),
                2,
            )

with open("utils/coco.txt") as f:
    class_list = f.read().split("\n")

Box_colours = []
for i in range(len(class_list)):
    R = random.randint(0, 255)
    G = random.randint(0, 255)
    B = random.randint(0, 255)
    Box_colours.append((B, G, R))

model = YOLO(r"3339model\weights\best.pt", "v8")


caps = [cv2.VideoCapture(i) for i in range(2)]  

for cap in caps:
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 100)

if any(not cap.isOpened() for cap in caps):
    print("Unable to open one or more webcams. Exiting...")
    #exit()
else:
    print("Starting your webcams...")

while True:
    
    imgs = [cap.read()[1] for cap in caps]

    if any(img is None for img in imgs):
        print("Unable to load frames. Exiting...")
        break

    for i, img in enumerate(imgs):
        Detect_obj = model.track(source=[img], conf=0.50, save=False)
        predictions(img, Detect_obj, Box_colours, class_list)
        cv2.imshow(f'CAM-{i}', img)
        
    if cv2.waitKey(1) == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
