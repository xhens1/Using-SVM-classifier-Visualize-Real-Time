import numpy as np
from scipy.ndimage import interpolation
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import joblib, cv2, os
from skimage import color
from HogDB import DB

DB().Create_CopyData_Table()


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


model = joblib.load('models/models.dat')
filename = [os.path.abspath('copydata/copy2-1.avi'), os.path.abspath('copydata/copy2-2.avi'), os.path.abspath('copydata/copy2-3.avi')]
DB_data = []

# real time person detection
for temp in range(0, len(filename)):
    cap = cv2.VideoCapture(filename[temp])
    while True:
        ret, frame = cap.read()
        if type(frame) == type(None):
            break

        image = imutils.resize(frame, width=min(450, frame.shape[0]))
        size = (64, 128)
        step_size = (10, 10)
        downscale = 1.25
        detections = []
        scale = 0
        # The current scale of the image

        for im_scaled in pyramid_gaussian(image, downscale=downscale):
            # The list contains detections at the current scale
            if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, size, step_size):
                if im_window.shape[0] != size[1] or im_window.shape[1] != size[0]:
                    continue
                im_window = color.rgb2gray(im_window)
                fd1 = hog(im_window, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
                fd = fd1.reshape(1, -1)
                pred = model.predict(fd)

                if pred == 1:
                    if model.decision_function(fd) > 0.5:
                        detections.append(
                            (int(x * (downscale ** scale)), int(y * (downscale ** scale)), model.decision_function(fd),
                             int(size[0] * (downscale ** scale)),
                             int(size[1] * (downscale ** scale))))
            scale += 1

        clone = image.copy()
        for (x_tl, y_tl, _, w, h) in detections:
            cv2.rectangle(image, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)

        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        sc = [score[0] for (x, y, score, w, h) in detections]
        print('sc:', sc)
        sc = np.array(sc)
        pick = non_max_suppression(rects, probs=sc, overlapThresh=0.2)
        # list에 append 하면 될듯
        DB_data.append(list(fd1))
        DB_data.append(len(pick))
        for (x1, y1, x2, y2) in pick:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(clone, 'Person : {:.2f}'.format(np.max(sc)), (x1 - 2, y1 - 2), 1, 1, (0, 122, 12), 1)
        cv2.imshow('Person Detection', clone)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# video 마다 리스트 해서 대표 영상 3개 정보 한번에 DB에 저장
videoHog = DB_data[0::2]
videpPersonNum = DB_data[1::2]
DB().Insert_Copy_Video_Data(videoHog, videpPersonNum)
