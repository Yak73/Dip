import numpy as np
import cv2 as cv

cap = cv.VideoCapture('circle.mp4')
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(5, 5),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

save_point = []
cntr = 0

fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('output_modified.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while 1:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points

    print(" BE4 if: len(p1) = {} len(p0) = {}".format(len(p1), len(p0)))
    if len(p1[st == 1]) < len(p0[st == 1]):
        p1 = p0
    print(" AFTER if len(p1) = {} len(p0) = {}".format(len(p1), len(p0)))

    good_new = p1[st != 0]
    good_old = p0[st != 0]

    if cntr == 100:
        x_basis = min([x for x, y in good_new])
        y_basis = [y for x, y in good_new if x == x_basis][0]
        for x, y in good_new:
            if x != x_basis and y != y_basis:
                save_point.append((x - x_basis, y - y_basis))

    if len(p1) != len(good_new) or good_new[0][0] == good_old[0][0]:
        try:
            x_basis = min([x for x, y in good_new])
        except ValueError:
            break
        y_basis = round([y for x, y in good_new if x == x_basis][0], 3)
        x_basis = round(x_basis, 3)

        point_new = np.ones((len(save_point) + 1, 2), dtype='float32', order='C')
        point_new[0] = np.array((x_basis, y_basis), order='C')
        n = 1
        for x, y in save_point:
            x_new = round(x_basis + x, 3)
            y_new = round(y_basis + y, 3)
            point_new[n] = np.array((x_new, y_new), order='C')
            n += 1
        good_new = point_new.copy()

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        print(a, b)
        frame = cv.circle(frame, (int(a), int(b)), 4, (0, 255, 0), -1)
    cv.imshow('frame', frame)
    if ret:
        frame = cv.flip(frame, 0)
        out.write(frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    cntr += 1
k = cv.waitKey(0)
cv.destroyAllWindows()
out.release()
cap.release()
