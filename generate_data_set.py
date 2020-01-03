# -*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time

def switch_channel(img):
    B, G, R = cv2.split(img)
    img_new = cv2.merge((R, G, B))
    return img_new

def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio):   # usually ratio = 0.25
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio / 2)
    padding_height = int(height * ratio / 2)
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height
    roi_x1 = 0 if roi_x1 < 0 else roi_x1
    roi_y1 = 0 if roi_y1 < 0 else roi_y1
    roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
    roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2
    return roi_x1, roi_y1, roi_x2, roi_y2

def crop_face(img, roi_x1, roi_y1, roi_x2, roi_y2, landmarks):
    face_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
    landmarks -= np.array([roi_x1, roi_y1])
    return face_img, landmarks


def check_rect_point(img, landmarks):
    '''
    :param img:
    :param landmarks:
    :return: bool, verify all the landmarks in the img then return Trur, else return False
    '''
    img_width, img_height = img.shape[0], img.shape[1]
    for each in landmarks:
        if each[0] < 0 or each[0] > img_height:
            print("超出边界")
            return False
        if each[1] < 0 or each[0] > img_width:
            print("超出边界")
            return False

    return True


def generate_dataset(img, img_path, rect, points):
    img_width, img_height = img.shape[0], img.shape[1]
    rect = [float(each) for each in rect]

    # all numbers in rect should be non-negative
    for i in range(len(rect)):
        rect[i] = 0 if rect[i] < 0 else rect[i]

    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()  # 反转Y坐标轴
    x1, y1 = rect[:2]
    x2, y2 = rect[2:]
    x1, y1, x2, y2 = expand_roi(x1, y1, x2, y2, img_height, img_width, 0.25)
    plt.imshow(img)
    # top line
    y_top = np.ones(1000) * y1
    x_top = np.linspace(x1, x2, 1000)
    ax.plot(x_top, y_top, c='g')

    # right line
    x_right = np.ones(1000) * x2
    y_right = np.linspace(y1, y2, 1000)
    ax.plot(x_right, y_right, c='g')

    # bottom line
    x_bottom = np.linspace(x1, x2, 1000)
    y_bottom = np.ones(1000) * y2
    ax.plot(x_bottom, y_bottom, c='g')

    # left line
    x_left = np.ones(1000) * x1
    y_left = np.linspace(y1, y2, 1000)
    ax.plot(x_left, y_left, c='g')

    # scatter key point
    scatter_x = [float(points[i]) for i in range(len(points)) if i % 2 == 0]
    scatter_y = [float(points[i]) for i in range(len(points)) if i % 2 == 1]
    landmarks = list(zip(scatter_x, scatter_y))
    ax.scatter(scatter_x, scatter_y, s=1, c='r', marker='.')
    # plt.title(name, fontsize='xx-large')
    plt.axis('off')
    # write to train.txt and test.txt
    rect = str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2)
    content = img_path + ' ' + rect


    # plt.show()
    ax.cla()
    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()  # 反转Y坐标轴
    # print('befor crop:', landmarks)
    face_img, landmarks = crop_face(img, int(x1), int(y1), int(x2), int(y2), landmarks)
    check_result = check_rect_point(face_img, landmarks)
    if not check_result:
        return
    landmarks_list = landmarks.reshape(1, -1).tolist()[0]
    for each in landmarks_list:
        content += ' ' + str(each)
    # print(content)
    random_num = random.randint(1, pro_train + pro_test)
    if random_num <= pro_train:
        train_fs.write(content + '\n')
    else:
        test_fs.write(content + '\n')
    plt.imshow(face_img)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=1, c='r', marker='.')
    plt.axis('off')
    # plt.show()
    ax.cla()

if __name__ == "__main__":
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH, "data")
    folder_list = ['I', 'II']
    TRAIN_PATH = os.path.join(ROOT_PATH, 'train.txt')
    pro_train = 5   # probability of train dataset
    TEST_PATH = os.path.join(ROOT_PATH, 'test.txt')
    pro_test = 1   # probability of test dataset
    train_fs = open(TRAIN_PATH, 'w')
    test_fs = open(TEST_PATH, 'w')
    fig, ax = plt.subplots()
    for each_foler in folder_list:
        FILE_PATH = os.path.join(DATA_PATH, each_foler, 'label.txt')
        with open(FILE_PATH) as fs:
            for each_line in fs:
                info = each_line.strip().split()
                name = info[0]     # img name
                rect = info[1:5]   # face rect
                points = info[5:]  # key point
                img_path = os.path.join(DATA_PATH, each_foler, name)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = switch_channel(img)
                    generate_dataset(img, img_path, rect, points)
                else:
                    continue
    train_fs.close()
    test_fs.close()