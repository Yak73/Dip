import os
import cv2
import numpy as np


def segmentation(type_sample="Training"):
    os.chdir(type_sample)
    img_list = os.listdir()
    for i, img in enumerate(img_list):
        if os.path.isdir(img):
            continue
        image = cv2.imread(img)
        circles = detect_circles(img)
        try:
            os.mkdir("segmented")
        except OSError:
            pass
        os.chdir("segmented")
        for j, c in enumerate(circles[0, :]):
            x_left = int(c[0]) - int(c[2]) - 1
            x_right = int(c[0]) + int(c[2]) + 1
            y_top = int(c[1]) - int(c[2]) - 1
            y_bottom = int(c[1]) + int(c[2]) + 1
            cropped = image[y_top:y_bottom, x_left:x_right]
            cv2.imwrite(str(i)+str(j) + ".jpg", cropped)
        os.chdir(os.getcwd() + "/..")
    os.chdir(os.getcwd() + "/..")


def detect_circles(image, show_bit=0):
    src = cv2.imread(image)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                               param1=100, param2=50, minDist=150,
                               minRadius=25, maxRadius=120)
    if not show_bit:
        return circles

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(src, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(src, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('circles', src)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    return circles
