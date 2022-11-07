import cv2
import os

# https://github.com/kipr/opencv/tree/master/data/haarcascades

def get_frame_from_video(video_name, frame_time, img_dir=None, img_name=None):
    """
    get a specific frame of a video by time in milliseconds
    :param video_name: video name
    :param frame_time: time of the desired frame (ms)
    :param img_dir: path which use to store output image
    :param img_name: name of output image
    :return: None
    """
    vidcap = cv2.VideoCapture(video_name)
    # Current position of the video file in milli-seconds.
    vidcap.set(cv2.CAP_PROP_POS_MSEC, frame_time - 1)
    # read(): Grabs, decodes and returns the next video frame
    assert vidcap.isOpened()
    success, image = vidcap.read()
    # if not os.path.exists(img_dir):
    #     os.makedirs(img_dir)
    if success:
        # cv2.imwrite(os.path.join(img_dir, img_name), image)

        # x, y, w, h = 50, 50, 80, 80
        # cv2.rectangle(image, (x, y, x + w, y + h), color=(0, 255, 0), thickness=1)
        # x, y, r = 200, 200, 100
        # cv2.circle(image, center=(x, y), radius=r, color=(0, 0, 255), thickness=1)
        cv2.imshow("frame-%s" % frame_time, image)
        cv2.waitKey()
    cv2.destroyAllWindows()
    return image


def camera_capture():
    cap = cv2.VideoCapture(0)  # 生成读取摄像头对象
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度  640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度  480
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率 30fps
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码  22
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    print(width, height, fps, fourcc)
    # 定义视频对象输出
    writer = cv2.VideoWriter("video/video_output.mp4", fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()  # 读取摄像头画面
        cv2.imshow('video', frame)  # 显示画面
        # 等待键盘输入，参数1表示延时1ms切换到下一帧，参数为0表示显示当前帧，相当于暂停
        key = cv2.waitKey(20)
        writer.write(frame)  # 视频保存
        if key == 27 or key == ord('q'):   # 按ESC或Q退出
            break
    cap.release()  # 释放摄像头
    writer.release()
    cv2.destroyAllWindows()  # 释放所有显示图像窗口


# def video_to_frames(path):
#     """
#     输入：path(视频文件的路径)
#     """
#     # VideoCapture视频读取类
#     # videoCapture = cv2.VideoCapture()
#     # videoCapture.open(path)
#     videoCapture = cv2.VideoCapture(path)
#     assert videoCapture.isOpened()
#     frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)    # 总帧数
#     img_list = []
#     for i in range(int(frames)):
#         ret, frame = videoCapture.read()  # 按帧读取视频
#         if ret is False:
#             break
#         if i % 4 == 0:
#             img_list.append(frame)   # 3-channel rgb
#     print("视频切帧完成！")
#     for i, img in enumerate(img_list):
#         cv2.imwrite(f'video_caps/{i}.jpg', img)
#     return img_list


def video_to_frames(path):
    """
    输入：path(视频文件的路径)
    """
    # VideoCapture视频读取类
    # videoCapture = cv2.VideoCapture()
    # videoCapture.open(path)
    videoCapture = cv2.VideoCapture(path)
    assert videoCapture.isOpened()
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
    fps = videoCapture.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    fourcc = int(videoCapture.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)    # 总帧数
    duration = frames / fps
    print(width, height, fps, fourcc, frames, duration)
    timeF = int(fps)     # 视频帧计数间隔频率
    i = 0
    n = 1
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()  # 按帧读取视频
        # 到视频结尾时终止
        if ret is False:
            break
        # 每隔timeF帧进行存储操作
        if n % timeF == 0:
            i += 1
            print('保存第 %s 张图像' % i)
            save_image_dir = os.path.join('video', '%s.jpg' % i)
            print('save_image_dir: ', save_image_dir)
            cv2.imwrite(save_image_dir, frame)   # 保存视频帧图像
        n += 1
        cv2.waitKey(1)  # 延时1ms
    videoCapture.release()  # 释放视频对象
    cv2.destroyAllWindows()


def face_detect_from_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载特征数据
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    # 进行检测人脸操作
    # faces = face_detector.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5)
    faces = face_detector.detectMultiScale(gray)
    # 得到的faces可能是多组x,y,h,w
    print("faces 的数值：", faces)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    cv2.imshow('result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def face_detect_from_video(video_path):
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    videoCapture = cv2.VideoCapture(video_path)
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()  # 按帧读取视频
        # 到视频结尾时终止
        if ret is False:
            break
        faces = face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.imshow('result', frame)
        key = cv2.waitKey(2)
        if ord('q') == key or 32 == key:
            break
    cv2.destroyAllWindows()
    videoCapture.release()

# video_to_frames('video/video_output.mp4')
# camera_capture()
# image = get_frame_from_video('video/video_output.mp4', 300)
# face_detect_from_video('video/lecture-short.mp4')

# video_to_frames('E:\GRID\\video\s1\\bbaf2n.mpg')
face_detect_from_video('E:\GRID\\video\s1\\bbaf2n.mpg')