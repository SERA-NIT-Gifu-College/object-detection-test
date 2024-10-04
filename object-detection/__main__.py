import cv2
import numpy as np

cap = cv2.VideoCapture('object-detection/data/video-resized.mp4')

objects = {
    'cone': cv2.imread('object-detection/data/cone.jpg')
}

while True:
    ret, src = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    for name, obj in objects.items():
        map_cc = cv2.matchTemplate(src, obj, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_xy = cv2.minMaxLoc(map_cc)
        print(max_v, max_xy)
        if max_v > 0.55:
            to_xy = (max_xy[0] + obj.shape[1], max_xy[1] + obj.shape[0])
            cv2.rectangle(src, max_xy, to_xy, (255,255,255), 4)
            print("object detected!")
        cv2.imshow('win_map_' + name, map_cc)
    cv2.imshow('win_src', src)
    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == 32:
        cv2.waitKey(0)

cv2.destroyAllWindows()
