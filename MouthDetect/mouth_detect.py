import os
import cv2
import dlib
import numpy as np

# dlib可以检测图像中的人脸，并且可以检测出人脸上的68个关键点，
# 其中后20个点表示了唇部的关键点，因此可以使用dlib检测人脸并通过嘴部关键点得到嘴部图像

def mouth_detect():
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    face_file = os.path.join(CURRENT_PATH, "video", "1.jpg")
    img = cv2.imread(face_file)
    # 给定输出规模shape=[width, height]
    shape = [100, 50]
    # 加载面部检测模型
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    # 检测面部并选择
    faces = detector(img, 1)
    face = faces[0]
    # 检测关键点并选择唇部
    points = predictor(img, face)
    mouth_points = np.array([(point.x, point.y) for point in points.parts()[48:]])
    # 截取唇部图像
    center = np.mean(mouth_points, axis=0).astype(int)
    rect = cv2.boundingRect(mouth_points)
    ratio = rect[2] / rect[3]
    if shape[0] / shape[1] > ratio:
      shape[0], shape[1] = shape[0] / shape[1] * rect[3], rect[3]
    else:
      shape[0], shape[1] = rect[2], shape[1] / shape[0] * rect[2]

    mouth_img = img[int(center[1] - shape[1] / 2): int(center[1] + shape[1] / 2),
                 int(center[0] - shape[0] / 2): int(center[0] + shape[0] / 2)]
    cv2.imwrite("mouth.png", mouth_img)


def mouth_detect2():
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    face_file = os.path.join(CURRENT_PATH, "video", "1.jpg")
    image = cv2.imread(face_file)
    PREDICTOR_FILE = os.path.join(CURRENT_PATH, "shape_predictor_68_face_landmarks.dat")
    normalize_ratio = None  # 缩放比例
    HORIZONTAL_PAD = 0.19  # 填充比例
    MOUTH_WIDTH = 100  # 图像固定大小 宽
    MOUTH_HEIGHT = 50  # 高
    # 人脸前端检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_FILE)
    dets = detector(image, 1)   # 人脸检测
    # 如果一张图像有多个人脸
    for k, d in enumerate(dets):
        # rec_image = cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), color=(0, 255, 0), thickness=2)
        # cv2.imwrite("face.png", rec_image)
        # 在人脸的基础上，检测68个特征点
        shape = predictor(image, d)
        mouth_points = [(part.x, part.y) for part in shape.parts()[48:]]  # 收集嘴唇坐标值
        # 保存嘴唇检测特征点图像
        # for i in mouth_points:
        #     cv2.circle(image, i, 10, (0, 0, 255), thickness=2)
        # cv2.imwrite("lip-feat-20.png", image)
        np_mouth_points = np.array(mouth_points)
        # 计算中心坐标
        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)
        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)
            # 计算缩放比例
            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)
        # 将原图像进行缩放
        # new_image_shape = (int(image.shape[0] * normalize_ratio), int(image.shape[1] * normalize_ratio))
        # new_resize_image = cv2.resize(image, new_image_shape, interpolation=cv2.INTER_NEAREST)  # cv2.INTER_CUBIC cv2.INTER_AREA cv2.INTER_NEAREST
        # 新的中心坐标值
        mouth_centroid_new = mouth_centroid * normalize_ratio
        # 计算边界
        mouth_l = int(mouth_centroid_new[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_new[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_new[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_new[1] + MOUTH_HEIGHT / 2)
        # 画出嘴唇的部分的框图
        # mouth_rec_new_image = cv2.rectangle(image, (mouth_l, mouth_t), (mouth_r, mouth_b), thickness=2,
        #                                    color=(0, 0, 255))
        # cv2.imwrite("lip.png", mouth_rec_new_image)
        mouth_clip = image[mouth_t:mouth_b, mouth_l:mouth_r]
        cv2.imwrite("lip.png", mouth_clip)


mouth_detect()
mouth_detect2()