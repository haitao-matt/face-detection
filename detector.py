from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

import runpy
import numpy as np
import os
import cv2
import time

from data import get_train_test_set, draw_picture
# from predict import predict

torch.set_default_tensor_type(torch.FloatTensor)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Backbone:
        # in_channel, out_channel, kernel_size, stride, padding
        # block 1
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2, 0)
        # block 2
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        # block 3
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        # block 4
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        # points branch
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)
        # common used
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # block 1
        # print('x input shape: ', x.shape)
        x = self.ave_pool(self.prelu1_1(self.conv1_1(x)))
        # print('x after block1 and pool shape should be 32x8x27x27: ', x.shape)     # good
        # block 2
        x = self.prelu2_1(self.conv2_1(x))
        # print('b2: after conv2_1 and prelu shape should be 32x16x25x25: ', x.shape) # good
        x = self.prelu2_2(self.conv2_2(x))
        # print('b2: after conv2_2 and prelu shape should be 32x16x23x23: ', x.shape) # good
        x = self.ave_pool(x)
        # print('x after block2 and pool shape should be 32x16x12x12: ', x.shape)
        # block 3
        x = self.prelu3_1(self.conv3_1(x))
        # print('b3: after conv3_1 and pool shape should be 32x24x10x10: ', x.shape)
        x = self.prelu3_2(self.conv3_2(x))
        # print('b3: after conv3_2 and pool shape should be 32x24x8x8: ', x.shape)
        x = self.ave_pool(x)
        # print('x after block3 and pool shape should be 32x24x4x4: ', x.shape)
        # block 4
        x = self.prelu4_1(self.conv4_1(x))
        # print('x after conv4_1 and pool shape should be 32x40x4x4: ', x.shape)

        # points branch
        ip3 = self.prelu4_2(self.conv4_2(x))
        # print('pts: ip3 after conv4_2 and pool shape should be 32x80x4x4: ', ip3.shape)
        ip3 = ip3.view(-1, 4 * 4 * 80)
        # print('ip3 flatten shape should be 32x1280: ', ip3.shape)
        ip3 = self.preluip1(self.ip1(ip3))
        # print('ip3 after ip1 shape should be 32x128: ', ip3.shape)
        ip3 = self.preluip2(self.ip2(ip3))
        # print('ip3 after ip2 shape should be 32x128: ', ip3.shape)
        ip3 = self.ip3(ip3)
        # print('ip3 after ip3 shape should be 32x42: ', ip3.shape)
        return ip3


def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    pts_criterion = criterion

    train_losses = []
    valid_losses = []

    for epoch_id in range(epoch):
        # monitor training loss
        ######################
        # training the model #
        ######################
        model.train()
        train_mean_pts_loss = 0.0
        train_batch_cnt = 0

        for batch_idx, batch in enumerate(train_loader):
            train_batch_cnt += 1
            img = batch['image']

            landmark = batch['landmarks']
            # print(landmark.size())
            # time.sleep(100)
            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            output_pts = model(input_img)

            # get loss
            loss = pts_criterion(output_pts, target_pts)
            loss *= 10000
            train_mean_pts_loss += loss.item()
            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print("#############target_pts################")
                print(target_pts * 112)
                print("#############target_pts################")
                print("#############output_pts################")
                print(output_pts * 112)
                print("#############output_pts################")
                print("#############output_pts - target_pts################")
                print((output_pts - target_pts) * 112)
                print("#############output_pts - target_pts################")
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                        epoch_id,
                        batch_idx * len(img),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()
                    )
                )
        train_mean_pts_loss /= train_batch_cnt
        train_losses.append(train_mean_pts_loss)
        ######################
        # validate the model #
        ######################
        valid_mean_pts_loss = 0.0

        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)

                valid_loss = pts_criterion(output_pts, target_pts)
                valid_loss *= 10000
                valid_mean_pts_loss += valid_loss.item()

            valid_mean_pts_loss /= valid_batch_cnt
            valid_losses.append(valid_mean_pts_loss)
            print('Valid: pts_loss: {:.6f}'.format(
                    valid_mean_pts_loss
                )
            )
        print('====================================================')
        # save model
        if args.save_model and (epoch_id + 1) % 1000 == 0:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id + 1) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return train_losses, valid_losses


def test(args, test_loader, model, criterion, optimizer, device):
    ######################
    # testing the model #
    ######################
    model.eval()
    pts_criterion = criterion
    test_mean_pts_loss = 0.0
    with torch.no_grad():
        test_batch_cnt = 0
        for test_batch_idx, batch in enumerate(test_loader):
            test_batch_cnt += 1
            test_img = batch['image']
            landmark = batch['landmarks']

            input_img = test_img.to(device)
            target_pts = landmark.to(device)
            output_pts = model(input_img)
            test_loss = pts_criterion(output_pts, target_pts)
            #change to 112 * 112 size
            test_loss *= 112
            test_mean_pts_loss += test_loss.item()
        test_mean_pts_loss /= test_batch_cnt
        print('Test: pts_loss: {:.6f}'.format(
            test_mean_pts_loss
        )
        )

    return test_mean_pts_loss


def predict(args, predict_loader, model, device):
    # save imgs
    if not os.path.exists(args.predict_directory):
        os.mkdir(args.predict_directory)

    ##############################
    # predict the predict_loader #
    ##############################
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(predict_loader):
            img = batch['image']
            img_name = batch['img_name']
            img_color = batch['img_color']
            input_img = img.to(device)
            output_pts = model(input_img)
            # change to 112 * 112
            output_pts *= 112
            draw_picture(args.predict_directory, img_name, output_pts, img_color)


def finetune(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    pts_criterion = criterion

    train_losses = []
    valid_losses = []

    for epoch_id in range(epoch):
        # monitor training loss
        ######################
        # funtuning the model #
        ######################
        model.train()
        train_mean_pts_loss = 0.0
        train_batch_cnt = 0
        print(train_loader.keys())
        for batch_idx, batch in enumerate(train_loader):
            train_batch_cnt += 1
            img = batch['image']


            landmark = batch['landmarks']

            input_img = img.to(device)
            target_pts = landmark.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            output_pts = model(input_img)

            # get loss
            loss = pts_criterion(output_pts, target_pts)
            loss *= 10000
            train_mean_pts_loss += loss.item()
            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print("#############target_pts################")
                print(target_pts * 112)
                print("#############target_pts################")
                print("#############output_pts################")
                print(output_pts * 112)
                print("#############output_pts################")
                print("#############output_pts - target_pts################")
                print((output_pts - target_pts) * 112)
                print("#############output_pts - target_pts################")
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                        epoch_id,
                        batch_idx * len(img),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()
                    )
                )
        train_mean_pts_loss /= train_batch_cnt
        train_losses.append(train_mean_pts_loss)
        ######################
        # validate the model #
        ######################
        valid_mean_pts_loss = 0.0

        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)

                valid_loss = pts_criterion(output_pts, target_pts)
                valid_loss *= 10000
                valid_mean_pts_loss += valid_loss.item()

            valid_mean_pts_loss /= valid_batch_cnt
            valid_losses.append(valid_mean_pts_loss)
            print('Valid: pts_loss: {:.6f}'.format(
                    valid_mean_pts_loss
                )
            )
        print('====================================================')
        # save model
        if args.save_model and epoch_id == epoch - 1:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return train_losses, valid_losses

def main_test():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--predict-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for predicting (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--predict-directory', type=str, default='predict_img',
                        help='predicted pictures are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='train, test, predict or finetune')
    parser.add_argument('--test-model', type=str, default='detector_epoch_1000.pt',
                        help='test model name')
    parser.add_argument('--predict-model', type=str, default='detector_epoch_1000.pt',
                        help='predict model name')
    args = parser.parse_args()
    ###################################################################################
    # print(args)
    torch.manual_seed(args.seed)
    # root_path
    root_path = os.getcwd()
    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    # cuda:0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    predict_loader = torch.utils.data.DataLoader(train_set, batch_size=args.predict_batch_size)   # predict_batch_size:1

    print('===> Building Model')
    # For single GPU
    model = Net().to(device)
    ####################################################################
    criterion_pts = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    optimizer_adam = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    ####################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        x = range(args.epochs)
        plt.plot(x, train_losses, color="r", linestyle="-", marker="o", linewidth=1, label="train")
        plt.plot(x, valid_losses, color="b", linestyle="-", marker="o", linewidth=1, label="val")
        plt.legend()
        plt.title('train and val loss vs. epoches')
        plt.ylabel('loss')
        plt.savefig("train and val loss vs epoches.jpg")
        plt.close('all')  # 关闭图 0
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        model_name = args.test_model
        model_path = os.path.join(root_path, "trained_models", model_name)
        if os.path.exists(model_path):
            # how to do test?
            model.load_state_dict(torch.load(model_path), map_location=torch.device(device))
            valid_losses = test(args, test_loader, model, criterion_pts, optimizer, device)
            print('====================================================')
        else:
            print("model not exists.")
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        # how to do finetune?
        model_name = args.predict_model
        model_path = os.path.join(root_path, "trained_models", model_name)

        if os.path.exists(model_path):
            # how to do test?
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            finetune_losses, valid_losses = finetune(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
            x = range(args.epochs)
            plt.plot(x, finetune_losses, color="r", linestyle="-", marker="o", linewidth=1, label="finetune")
            plt.plot(x, valid_losses, color="b", linestyle="-", marker="o", linewidth=1, label="val")
            plt.legend()
            plt.title('finetune and val loss vs. epoches')
            plt.ylabel('loss')
            plt.savefig("finetune and val loss vs epoches.jpg")
            plt.close('all')  # 关闭图 0
            print('====================================================')
        else:
            print("model not exists.")
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        # how to do predict?
        model_name = args.predict_model
        model_path = os.path.join(root_path, "trained_models", model_name)

        if os.path.exists(model_path):
            # how to do test?
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            predict(args, predict_loader, model, device)
            print('====================================================')
        else:
            print("model not exists.")

if __name__ == '__main__':
    main_test()