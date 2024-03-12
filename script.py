import cv2
import numpy as np

img = cv2.imread("D:/PycharmProjects\\Work_landmark\\Data\\300W\\frontal_train\\815038_2.jpg") # 分割出来的人脸大头照
# case 1 : 读取npy
# npy_data = np.load("D:/PycharmProjects\\Work_landmark\\Data\\300W\\frontal_train_label\\815038_2.jpg.npy")
# npy_data = np.array(npy_data, dtype=np.int32)
# print(npy_data)
# for i in range(68):
#     cv2.circle(img, (npy_data[i, 0], npy_data[i, 1]), 1, (0, 0, 255), 2)
# cv2.imshow('case1', img)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

img = cv2.imread("D:/PycharmProjects\\Work_landmark\\Data\\300W\\afw\\815038_2.jpg") # 整张图片
# case 2 : 读取pts
pts_data = np.genfromtxt("D:/PycharmProjects\\Work_landmark\\Data\\300W\\afw\\815038_2.pts", skip_header=3,
                                       skip_footer=1, delimiter=' ') - 1.0
pts_data = np.array(pts_data, dtype=np.int32)
print(pts_data)
for i in range(68):
    cv2.circle(img, (pts_data[i, 0], pts_data[i, 1]), 1, (0, 0, 255), 2)
cv2.imshow('case2', img)
cv2.waitKey(0)

cv2.destroyAllWindows()