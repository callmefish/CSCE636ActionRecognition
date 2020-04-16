import cv2

import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
from glob import glob

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import test_spatial_dataloader
from dataloader import test_motion_dataloader

from utils import *
from network import *
import math
import json
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 添加命令行指令
parser = argparse.ArgumentParser(description='video test for two stream')
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')


def main(start_frame):
    global arg
    global whole_dic
    arg = parser.parse_args()

    # spatial Prepare DataLoader
    spatial_data_loader = test_spatial_dataloader(
        BATCH_SIZE=arg.batch_size,
        num_workers=8,
        path='/home/yzy20161103/csce636_project/project/record/temp_chunk/',
    )
    spatial_test_loader = spatial_data_loader.run()
    # spatial Model
    spatial_model = Spatial_CNN(
        lr=arg.lr,
        batch_size=arg.batch_size,
        resume='/home/yzy20161103/csce636_project/project/record/spatial_res_3_100/model_best.pth.tar',
        start_epoch=0,
        evaluate='evaluate',
        test_loader=spatial_test_loader,
        start_frame=start_frame,
    )
    spatial_model.run()

    # motion prepare dataloader
    motion_data_loader = test_motion_dataloader(
        BATCH_SIZE=arg.batch_size,
        num_workers=8,
        in_channel=10,
        path='/home/yzy20161103/csce636_project/project/record/temp_opf/'
    )
    motion_test_loader = motion_data_loader.run()
    # motion Model
    motion_model = Motion_CNN(
        test_loader=motion_test_loader,
        start_epoch=0,
        resume='/home/yzy20161103/csce636_project/project/record/motion_res_3_100/model_best.pth.tar',
        evaluate='evaluate',
        lr=arg.lr,
        batch_size=arg.batch_size,
        channel=10 * 2,
        start_frame=start_frame,
    )
    # Training
    motion_model.run()


class Spatial_CNN():
    global whole_dic

    def __init__(self, lr, batch_size, resume, start_epoch, evaluate, test_loader, start_frame):
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.start_epoch = start_epoch
        self.evaluate = evaluate
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.start_frame = start_frame

    def build_model(self):
        # build model
        self.model = resnet101(pretrained=True, channel=3).cuda()
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()

    def validate_1epoch(self):
        batch_time = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (keys, data, label) in enumerate(progress):
                data = data.cuda()
                # compute output
                output = self.model(data)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # Calculate video level prediction
                preds = output.data.cpu().numpy()
                nb_data = preds.shape[0]
                for j in range(nb_data):
                    videoName = keys[j].split('/', 1)[0] + '_' + str(self.start_frame)
                    if videoName not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[videoName] = preds[j, :]
                        whole_dic_spatial[videoName] = preds[j, :]
                    else:
                        self.dic_video_level_preds[videoName] += preds[j, :]
                        whole_dic_spatial[videoName] += preds[j, :]
        return


class Motion_CNN():
    def __init__(self, lr, batch_size, resume, start_epoch, evaluate, test_loader, channel,
                 start_frame):
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.start_epoch = start_epoch
        self.evaluate = evaluate
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.channel = channel
        self.start_frame = start_frame
        self.img_rows = 224
        self.img_cols = 224

    def build_model(self):
        # build model
        self.model = resnet101(pretrained=True, channel=self.channel).cuda()
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()

    def validate_1epoch(self):
        batch_time = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (keys, data, label) in enumerate(progress):
                data = data.cuda()
                # compute output
                output = self.model(data)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # Calculate video level prediction
                preds = output.data.cpu().numpy()
                nb_data = preds.shape[0]
                for j in range(nb_data):
                    videoName = keys[j].split('/', 1)[0] + '_' + str(self.start_frame)
                    if videoName not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[videoName] = preds[j, :]
                        whole_dic_motion[videoName] = preds[j, :]
                    else:
                        self.dic_video_level_preds[videoName] += preds[j, :]
                        whole_dic_motion[videoName] += preds[j, :]
        return


def cal_for_frames(video_path, video_name, flow_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        # flow.append(tmp_flow)
        prev = curr
        if not os.path.exists(os.path.join(flow_path, video_name + '_u')):
            os.mkdir(os.path.join(flow_path, video_name + '_u'))
        cv2.imwrite(os.path.join(flow_path, video_name + '_u', "{:06d}.jpg".format(i+1)), tmp_flow[:, :, 0])
        if not os.path.exists(os.path.join(flow_path, video_name + '_v')):
            os.mkdir(os.path.join(flow_path, video_name + '_v'))
        cv2.imwrite(os.path.join(flow_path, video_name + '_v', "{:06d}.jpg".format(i+1)), tmp_flow[:, :, 1])

    return


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


def extract_flow(video_path, video_name, flow_path):
    cal_for_frames(video_path, video_name, flow_path)
    print('complete:' + flow_path + video_name)
    return


if __name__ == '__main__':
    file_path = '/home/yzy20161103/csce636_project/project/record/sample_video/sample_video_01(S).mp4'
    video_title = file_path.split('/')[-1][:-4]
    print(video_title)
    cap = cv2.VideoCapture(file_path)
    # file_path是文件的绝对路径，防止路径中含有中文时报错，需要解码
    if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
        # get方法参数按顺序对应下表（从0开始编号)
        rate = cap.get(5)  # 帧速率
        FrameNumber = cap.get(7)  # 视频文件的帧数
        duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，秒
        print(duration)
    rgb_outPutDirName = '/home/yzy20161103/csce636_project/project/record/temp_chunk/'
    opf_outPutDirName = '/home/yzy20161103/csce636_project/project/record/temp_opf/'

    if not os.path.exists(rgb_outPutDirName):
        # 如果文件目录不存在则创建目录
        os.makedirs(rgb_outPutDirName)
    if not os.path.exists(opf_outPutDirName):
        # 如果文件目录不存在则创建目录
        os.makedirs(opf_outPutDirName)

    frame = 1
    index = 0
    whole_dic_spatial = {}
    whole_dic_motion = {}

    while True:
        res, image = cap.read()
        if not res:
            print('not res , not image')
            if len(os.listdir(rgb_outPutDirName)) > 1:
                extract_flow(rgb_outPutDirName, 'v_temp_opf', opf_outPutDirName)
                StartFrame = frame - len(os.listdir(rgb_outPutDirName))
                main(StartFrame)
                shutil.rmtree(rgb_outPutDirName)
                shutil.rmtree(opf_outPutDirName)
                os.makedirs(opf_outPutDirName)
            break
        else:
            cv2.imwrite(rgb_outPutDirName + 'frame_' + str(index + 1).zfill(6) + '.jpg', image)
            frame += 1
            index += 1
            if (frame - 1) % 100 == 0:
                extract_flow(rgb_outPutDirName, 'v_temp_opf', opf_outPutDirName)
                StartFrame = frame - 101
                main(StartFrame)
                # break
                shutil.rmtree(rgb_outPutDirName)
                os.makedirs(rgb_outPutDirName)
                shutil.rmtree(opf_outPutDirName)
                os.makedirs(opf_outPutDirName)
                index = 0

    print('finish video analysis')
    cap.release()
    time_label = {'Slipping': []}
    whole_dic = {}
    
    for key in whole_dic_spatial.keys():
        new_key = 'temp_opf_' + key.split('_')[2]
        whole_dic[key] = whole_dic_spatial[key] + whole_dic_motion[new_key]

    list_key = list(whole_dic.keys())
    x = []
    y = []
    for i in range(len(list_key)):
        current_time = int(list_key[i].split('_')[2]) / rate
        x.append(current_time)
        # softmax
        value = whole_dic[list_key[i]]
        prec = math.exp(value[0]) / (math.exp(value[0]) + math.exp(value[1]))
        y.append(prec)
        time_label['Slipping'].append([current_time, prec])
    print('time_label : {}'.format(time_label))
    # os.makedirs('/home/yzy20161103/two-stream-action-recognition/record/time_label_data.json')
    json_str = json.dumps(time_label)
    with open('/home/yzy20161103/csce636_project/project/time_label_data_' + video_title + '.json', 'w') as json_file:
        json_file.write(json_str)
    plt.plot(x, y, linewidth=3, color='blue')
    plt.xlabel('time/s')
    plt.ylabel('Slipping')
    plt.savefig('/home/yzy20161103/csce636_project/project/result_' + video_title + '.png')
    plt.show()
