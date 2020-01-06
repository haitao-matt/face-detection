# -*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time


negsample_ratio = 0.3  # if the positive sample's iou > this ratio, we neglect it's negative samples
neg_gen_thre = 100
random_times = 3
random_border = 10
expand_ratio = 0.25


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
            # print("超出边界")
            return False
        if each[1] < 0 or each[0] > img_width:
            # print("超出边界")
            return False

    return True


def get_iou(rect1, rect2):
    # rect: 0-4: x1, y1, x2, y2
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    width1 = right1 - left1 + 1
    height1 = bottom1 - top1 + 1

    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    width2 = right2 - left2 + 1
    height2 = bottom2 - top2 + 1

    w_left = max(left1, left2)
    h_left = max(top1, top2)
    w_right = min(right1, right2)
    h_right = min(bottom1, bottom2)
    inner_area = max(0, w_right - w_left + 1) * max(0, h_right - h_left + 1)
    #print('wleft: ', w_left, '  hleft: ', h_left, '    wright: ', w_right, '    h_right: ', h_right)

    box1_area = width1 * height1
    box2_area = width2 * height2
    #print('inner_area: ', inner_area, '   b1: ', box1_area, '   b2: ', box2_area)
    iou = float(inner_area) / float(box1_area + box2_area - inner_area)
    return iou


def check_iou(rect1, rect2):
    # rect: 0-4: x1, y1, x2, y2
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    width1 = right1 - left1 + 1
    height1 = bottom1 - top1 + 1

    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    width2 = right2 - left2 + 1
    height2 = bottom2 - top2 + 1

    w_left = max(left1, left2)
    h_left = max(top1, top2)
    w_right = min(right1, right2)
    h_right = min(bottom1, bottom2)
    inner_area = max(0, w_right - w_left + 1) * max(0, h_right - h_left + 1)
    # print('wleft: ', w_left, '  hleft: ', h_left, '    wright: ', w_right, '    h_right: ', h_right)

    box1_area = width1 * height1
    box2_area = width2 * height2
    # print('inner_area: ', inner_area, '   b1: ', box1_area, '   b2: ', box2_area)
    iou = float(inner_area) / float(box1_area + box2_area - inner_area)
    return iou


def generate_random_crops(shape, rects, random_times):
    neg_gen_cnt = 0
    img_h = shape[0]
    img_w = shape[1]
    rect_wmin = img_w   # + 1
    rect_hmin = img_h   # + 1
    rect_wmax = 0
    rect_hmax = 0
    num_rects = len(rects)
    for rect in rects:
        w = rect[2] - rect[0] + 1
        h = rect[3] - rect[1] + 1
        if w < rect_wmin:
            rect_wmin = w
        if w > rect_wmax:
            rect_wmax = w
        if h < rect_hmin:
            rect_hmin = h
        if h > rect_hmax:
            rect_hmax = h
    random_rect_cnt = 0
    random_rects = []
    while random_rect_cnt < num_rects * random_times and neg_gen_cnt < neg_gen_thre:
        neg_gen_cnt += 1
        if img_h - rect_hmax - random_border > 0:
            top = np.random.randint(0, img_h - rect_hmax - random_border)
        else:
            top = 0
        if img_w - rect_wmax - random_border > 0:
            left = np.random.randint(0, img_w - rect_wmax - random_border)
        else:
            left = 0
        rect_wh = np.random.randint(min(rect_wmin, rect_hmin), max(rect_wmax, rect_hmax) + 1)
        rect_randw = np.random.randint(-3, 3)
        rect_randh = np.random.randint(-3, 3)
        right = left + rect_wh + rect_randw - 1
        bottom = top + rect_wh + rect_randh - 1

        good_cnt = 0
        for rect in rects:
            img_rect = [0, 0, img_w - 1, img_h - 1]
            rect_img_iou = get_iou(rect, img_rect)
            if rect_img_iou > negsample_ratio:
                random_rect_cnt += random_times
                break
            random_rect = [left, top, right, bottom]
            iou = get_iou(random_rect, rect)

            if iou < 0.2:
                # good thing
                good_cnt += 1
            else:
                # bad thing
                break

        if good_cnt == num_rects:
            # print('random rect: ', random_rect, '   rect: ', rect)
            _iou = check_iou(random_rect, rect)

            # print('iou: ', iou, '   check_iou: ', _iou)
            # print('\n')
            random_rect_cnt += 1
            random_rects.append(random_rect)
    return random_rects


def draw_rect(ax, x1, y1, x2, y2, color):
    # top line
    y_top = np.ones(1000) * y1
    x_top = np.linspace(x1, x2, 1000)
    ax.plot(x_top, y_top, c=color)

    # right line
    x_right = np.ones(1000) * x2
    y_right = np.linspace(y1, y2, 1000)
    ax.plot(x_right, y_right, c=color)

    # bottom line
    x_bottom = np.linspace(x1, x2, 1000)
    y_bottom = np.ones(1000) * y2
    ax.plot(x_bottom, y_bottom, c=color)

    # left line
    x_left = np.ones(1000) * x1
    y_left = np.linspace(y1, y2, 1000)
    ax.plot(x_left, y_left, c=color)


# generate_dataset(img, img_path, rect, points):
def generate_dataset(result, random_times):
    img_path = result['name']
    # print(img_path)
    img = cv2.imread(img_path)
    img = switch_channel(img)
    rects = result['rect']
    landmarks_ori = result['points']
    # print('points', landmarks)
    img_width, img_height = img.shape[0], img.shape[1]
    random_rects = generate_random_crops((img_width, img_height), rects, random_times)
    plt.imshow(img)
    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    # ax.invert_yaxis()  # 反转Y坐标轴

    for each_rect in range(len(rects)):
        rect = rects[each_rect]
        points = landmarks_ori[each_rect]
        # all numbers in rect should be non-negative
        for i in range(len(rect)):
            rect[i] = 0 if rect[i] < 0 else rect[i]

        x1, y1 = rect[:2]
        x2, y2 = rect[2:]
        x1, y1, x2, y2 = expand_roi(x1, y1, x2, y2, img_height, img_width, expand_ratio)
        # drwa rect
        draw_rect(ax, x1, y1, x2, y2, 'r')

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

        face_img, landmarks = crop_face(img, int(x1), int(y1), int(x2), int(y2), landmarks)
        check_result = check_rect_point(face_img, landmarks)
        if not check_result:
            # plt.close()
            # ax.cla()
            # return
            continue

        landmarks_list = landmarks.reshape(1, -1).tolist()[0]
        for each in landmarks_list:
            content += ' ' + str(each)
        content += ' 1'
        random_num = random.randint(1, ratio_train + ratio_test + ratio_predict)
        if random_num <= ratio_train:
            train_fs.write(content + '\n')
        elif random_num <= (ratio_train + ratio_test):
            test_fs.write(content + '\n')
        else:
            predict_fs.write(content + '\n')
        # plt.imshow(face_img)

        # ax.scatter(landmarks[:, 0], landmarks[:, 1], s=1, c='r', marker='.')
        # plt.axis('off')
        # plt.show()
        # ax.cla()
    for each in random_rects:
        draw_rect(ax, each[0], each[1], each[2], each[3], 'g')
        rect_coordinate = str(each[0]) + ' ' + str(each[1]) + ' ' + str(each[2]) + ' ' + str(each[3])
        contents = img_path + ' ' + rect_coordinate + ' ' + '0'
        # print(contents)
        random_num = random.randint(1, ratio_train + ratio_test + ratio_predict)
        if random_num <= ratio_train:
            train_fs.write(contents + '\n')
        elif random_num <= (ratio_train + ratio_test):
            test_fs.write(contents + '\n')
        else:
            predict_fs.write(contents + '\n')

    # plt.show()
    ax.cla()

if __name__ == "__main__":
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH, "data")
    folder_list = ['I', 'II']
    TRAIN_PATH = os.path.join(ROOT_PATH, 'train.txt')
    ratio_train = 5   # probability ratio of train dataset
    TEST_PATH = os.path.join(ROOT_PATH, 'test.txt')
    ratio_test = 1   # probability ratio of test dataset
    PREDICT_PATH = os.path.join(ROOT_PATH, 'predict.txt')
    ratio_predict = 1 # probability ratio of predict dataset
    train_fs = open(TRAIN_PATH, 'w')
    test_fs = open(TEST_PATH, 'w')
    predict_fs = open(PREDICT_PATH, 'w')
    fig, ax = plt.subplots()
    print("Begin generating datasets ...")
    for each_foler in folder_list:
        FILE_PATH = os.path.join(DATA_PATH, each_foler, 'label.txt')
        with open(FILE_PATH) as fs:
            img_name = ''
            result = {}
            for each_line in fs:
                info = each_line.strip().split()
                name = info[0]     # img name
                rect = info[1:5]   # face rect
                points = info[5:]  # key point
                img_path = os.path.join(DATA_PATH, each_foler, name)
                if os.path.exists(img_path):
                    if img_name == '':
                        img_name = name
                        result['name'] = img_path
                        result['rect'] = [list(map(float, rect))]
                        result['points'] = [list(map(float, points))]
                    elif img_name == name:
                        result['rect'].append(list(map(float, rect)))
                        result['points'].append(list(map(float, points)))
                    elif img_name != name:
                        # 调用处理函数
                        generate_dataset(result, random_times)
                        img_name = name
                        result = {}
                        result['name'] = img_path
                        result['rect'] = [list(map(float, rect))]
                        result['points'] = [list(map(float, points))]
                    # img = cv2.imread(img_path)
                    # img = switch_channel(img)
                    # generate_dataset(img, img_path, rect, points)
                else:
                    continue
            generate_dataset(result, random_times)
    print("Finish generating datasets ...")
    train_fs.close()
    test_fs.close()
    predict_fs.close()