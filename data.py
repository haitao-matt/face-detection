import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools
import matplotlib.pyplot as plt
import os
import time
import random
import cv2
import math


folder_list = ['I', 'II']
train_boarder = 112


unloader = transforms.ToPILImage()


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels, mean, std


def landmarks_norm(landmarks):
    landmarks /= 112
    return landmarks


def reverse_channel_norm(img, mean, std):
    pixels = img * (std + 0.0000001) + mean
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]

    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks, img_name, img_color = sample['image'], sample['landmarks'], sample['img_name'], \
                                                sample['img_color']
        image, mean, std = channel_norm(image)

        return {'image': image,
                'landmarks': landmarks,
                'mean': mean,
                'std': std,
                'img_name': img_name,
                'img_color': img_color
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks, mean, std, img_name, img_color = sample['image'], sample['landmarks'], sample['mean'],\
                                                           sample['std'], sample['img_name'], sample['img_color']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks),
                'mean': mean,
                'std': std,
                'img_name': img_name,
                'img_color': img_color
                }


class RandomHorizontalFlip(object):
    """
    Horizontal flip the img and corresponding coordinates
    :return: img after horizontal flip and corresponding coordinates
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, landmarks, img_name, img_color = sample['image'], sample['landmarks'], sample['img_name'],\
                                                sample['img_color']
        image_resize = np.asarray(
            image.resize((train_boarder, train_boarder), Image.BILINEAR),
            dtype=np.float32)  # Image.ANTIALIAS)
        image_color_resize = np.asarray(
            img_color.resize((train_boarder, train_boarder), Image.BILINEAR),
            dtype=np.float32)  # Image.ANTIALIAS)
        height, width = image_resize.shape[0], image_resize.shape[1]
        # angle_list = [-10, -30, -60, -90, 10, 30, 60, 90]
        angle_limit = 8
        if random.random() < self.p:
            angle = random.randint(-angle_limit, angle_limit)
            M = cv2.getRotationMatrix2D((image_resize.shape[1] / 2, image_resize.shape[0] / 2), angle, 1)
            img_rotate = cv2.warpAffine(image_resize, M, (image_resize.shape[1], image_resize.shape[0]))
            img_color_rotate = cv2.warpAffine(image_color_resize, M, (image_resize.shape[1], image_resize.shape[0]))

            x = list(map(float, landmarks[0::2]))
            y = list(map(float, landmarks[1::2]))
            points = list(zip(x, y))
            landmarks = []
            for i in range(len(points)):
                x1 = points[i][0]
                y1 = points[i][1]
                x2 = width / 2
                y2 = height / 2
                y = (y1 - y2) * math.sin(math.pi / 180 * angle) + (x1 - x2) * math.cos(math.pi / 180 * angle) + y2
                x = (y1 - y2) * math.cos(math.pi / 180 * angle) - (x1 - x2) * math.sin(math.pi / 180 * angle) + x2
                # points[i] = (y, x)
                landmarks.extend([y, x])

            landmarks = np.array(landmarks).astype(np.float32)

            return {'image': img_rotate,
                    'landmarks': landmarks,
                    'img_name': img_name,
                    'img_color': img_color_rotate
                    }
        return {'image': image_resize,
                'landmarks': landmarks,
                'img_name': img_name,
                'img_color': image_color_resize
                }


def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            RandomHorizontalFlip(),  # randomly flip
            Normalize(),                # do channel normalization
            ToTensor()]                 # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            RandomHorizontalFlip(),
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase, transform=tsfm)
    return data_set


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, phase, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        img_color = Image.open(img_name)
        # channels = len(img_color.size)

        img = img_color.convert('L')
        img_crop = img.crop(tuple(rect))
        img_color = img_color.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)
        # get img name
        img_name = os.path.basename(img_name)
        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:
        img_crop_width, img_crop_height = img_crop.size
        landmarks[0::2] *= train_boarder / img_crop_width
        landmarks[1::2] *= train_boarder / img_crop_height

        sample = {'image': img_crop, 'landmarks': landmarks, 'img_name': img_name, 'img_color': img_color}
        sample = self.transform(sample)

        if __name__ == "__main__":
            sample['original_image'] = img_color  #img_color为彩图
            sample['original_shape'] = img_crop.size  #img_color的大小和img_crop的大小一致
        else:
            sample['landmarks'] = landmarks_norm(sample['landmarks'])  #坐标归一化
        # if the original img is not color, then del img_color key
        if self.phase != 'predict':
            del sample['img_color']
        return sample


def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    predict_set = load_data('predict')
    return train_set, valid_set, predict_set


def draw_picture(save_directory, img_name, landmarks, img_color):
    img_color = img_color.squeeze(0)
    img_color = np.array(img_color).astype(np.float32)
    img_name = img_name[0]
    img_name = img_name.strip().split('.')[0] + '_crop_'

    for num in range(1, 100):
        save_name = img_name
        save_name += "%02d" % num + '.jpg'
        path = os.path.join(save_directory, save_name)
        if os.path.exists(path):
            continue
        else:
            break

    img_color = Image.fromarray(np.uint8(img_color))

    ## 请画出人脸crop以及对应的landmarks
    landmarks = landmarks.squeeze(0).tolist()
    # print(landmarks)

    scatter_x = list(map(int, landmarks[0::2]))
    scatter_y = list(map(int, landmarks[1::2]))

    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()  # 反转Y坐标轴
    plt.imshow(img_color)
    plt.title('%s\n112 * 112' % save_name)
    plt.axis('off')
    ax.scatter(scatter_x, scatter_y, s=2, c='r', marker='.')
    plt.savefig("%s" % path)
    # plt.show()
    ax.cla()


if __name__ == '__main__':
    predict_set = load_data('predict')
    for i in range(1, len(predict_set)):
        sample = predict_set[i]
        original_image, original_shape, mean, std, img_color = sample['original_image'], sample['original_shape'],\
                                                         sample['mean'], sample['std'], sample['img_color']
        img_crop_width, img_crop_height = original_shape

        img_color = Image.fromarray(np.uint8(img_color))
        landmarks = sample['landmarks']
        ## 请画出人脸crop以及对应的landmarks
        # please complete your code under this blank
        # img = img.numpy().transpose(1, 2, 0).astype(np.uint8)
        landmarks = landmarks.tolist()
        scatter_x = list(map(int, landmarks[0::2]))
        scatter_y = list(map(int, landmarks[1::2]))
        points = list(zip(scatter_x, scatter_y))

        plt.subplot(121)
        ax = plt.gca()  # 获取到当前坐标轴信息
        ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
        ax.invert_yaxis()  # 反转Y坐标轴
        plt.imshow(img_color)
        plt.title('image\n112 * 112')
        plt.axis('off')
        ax.scatter(scatter_x, scatter_y, s=2, c='r', marker='.')
        plt.subplot(122)
        plt.imshow(original_image)
        plt.title('original image\n{} * {}'.format(img_crop_width, img_crop_height))
        plt.axis('off')
        plt.show()
        ax.cla()