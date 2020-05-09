import cv2
import numpy as np
import pickle
import os
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
from test_spatial_dataloader import *
from test_motion_dataloader import *
from utils import *
from network import *
import math
import json
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 添加命令行指令
parser = argparse.ArgumentParser(description='video test for two stream')
parser.add_argument('--batch-size', default=19, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--video_name', type=str, default='csce636test/6700_2.mp4')
arg = parser.parse_args()
rgb_whole_pred = {}
opf_whole_pred = {}

def main(start_frame):
    # spatial Prepare DataLoader
    spatial_data_loader = test_spatial_dataloader(
        BATCH_SIZE=arg.batch_size,
        num_workers=8,
        path='record/temp_chunk/',
    )
    spatial_test_loader = spatial_data_loader.run()
    spatial_model = MODEL(
        lr=arg.lr,
        batch_size=arg.batch_size,
        # resume='best_model_475/spatial/model_best.pth.tar',
        resume='555_rgb_model_best.pth.tar',
        start_epoch=0,
        evaluate='evaluate',
        test_loader=spatial_test_loader,
        channel=3,
        start_frame=start_frame,
    )
    spatial_model.run()

    # motion prepare dataloader
    motion_data_loader = test_motion_dataloader(
        BATCH_SIZE=arg.batch_size,
        num_workers=8,
        in_channel=10,
        path='record/temp_opf/'
    )
    motion_test_loader = motion_data_loader.run()
    motion_model = MODEL(
        test_loader=motion_test_loader,
        start_epoch=0,
        resume='475_opt_model_best.pth.tar',
        # resume='555_opt_model_best.pth.tar',
        evaluate='evaluate',
        lr=arg.lr,
        batch_size=arg.batch_size,
        channel=10 * 2,
        start_frame=start_frame,
    )
    motion_model.run()


class MODEL():
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

    def build_model(self):
        self.model = resnet101(pretrained=True, channel=self.channel).cuda()
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
        self.model.eval()
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (keys, data, label) in enumerate(progress):
                data = data.cuda()
                output = self.model(data)
                batch_time.update(time.time() - end)
                end = time.time()
                preds = output.data.cpu().numpy()
                if self.channel == 20:
                    opf_whole_pred[str(self.start_frame)] = np.sum(preds, axis=0)
                else:
                    rgb_whole_pred[str(self.start_frame)] = np.sum(preds, axis=0)
        return


def cal_for_frames(video_path, video_name, flow_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
    prev = cv2.UMat(cv2.imread(frames[0]))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.UMat(cv2.imread(frame_curr))
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
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
    flow = cv2.UMat.get(flow)
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

def softmax(data):
    data_exp = np.exp(data)
    return data_exp / np.sum(data_exp)


if __name__ == '__main__':
    file_path = arg.video_name
    video_title = file_path.split('/')[-1][:-4]
    print(video_title)
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():  
        rate = cap.get(5)  
        FrameNumber = cap.get(7)  
        duration = FrameNumber / rate
        width = cap.get(3)
        height = cap.get(4)  
        print(duration)
    rgb_outPutDirName = 'record/temp_chunk/'
    opf_outPutDirName = 'record/temp_opf/'

    if not os.path.exists(rgb_outPutDirName):
        os.makedirs(rgb_outPutDirName)
    if not os.path.exists(opf_outPutDirName):
        os.makedirs(opf_outPutDirName)

    frame = 1
    index = 0
    

    while True:
        res, image = cap.read()
        if not res:
            print('not res , not image')
            if index > 27:
                extract_flow(rgb_outPutDirName, 'v_temp_opf', opf_outPutDirName)
                StartFrame = frame - len(os.listdir(rgb_outPutDirName))
                main(StartFrame)
            
            shutil.rmtree(rgb_outPutDirName)
            shutil.rmtree(opf_outPutDirName)    
            break
        else:
            if width < height:
                pad = int((height - width) // 2 + 1)
                image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
            image = cv2.resize(image, (342, 256))
            cv2.imwrite(rgb_outPutDirName + 'frame_' + str(index + 1).zfill(6) + '.jpg', image)
            frame += 1
            index += 1
            if (frame - 1) % 30 == 0:
                extract_flow(rgb_outPutDirName, 'v_temp_opf', opf_outPutDirName)
                StartFrame = frame - 31
                main(StartFrame)
                # break
                shutil.rmtree(rgb_outPutDirName)
                os.makedirs(rgb_outPutDirName)
                shutil.rmtree(opf_outPutDirName)
                os.makedirs(opf_outPutDirName)
                index = 0

    print('finish video analysis')
    cap.release()
    time_lable = {}

    fig_x = []
    fig_y = []
    fig_y_rgb = []
    fig_y_opf = []
    for key in list(rgb_whole_pred.keys()):
        cur_time = float(key)/rate
        new_key = str(float('%.3f'%cur_time))
        new_value = softmax(rgb_whole_pred[key] + 1 * opf_whole_pred[key]).tolist()
        rgb_value = softmax(rgb_whole_pred[key]).tolist()
        opf_value = softmax(opf_whole_pred[key]).tolist()
        time_lable[new_key] = new_value[0]
        fig_x.append(cur_time)
        fig_y.append(new_value[0])
        fig_y_rgb.append(rgb_value[0])
        fig_y_opf.append(opf_value[0])
    


    fig_x_1 = fig_x[:1]
    fig_y_1 = fig_y[:1]
    fig_y_rgb_1 = fig_y_rgb[:1]
    fig_y_opf_1 = fig_y_opf[:1]
    
    for i in range(1, len(fig_x)):
        fig_x_1.append(fig_x[i]-0.001)
        fig_x_1.append(fig_x[i])
        fig_y_1.append(fig_y_1[-1])
        fig_y_1.append(fig_y[i])
        fig_y_rgb_1.append(fig_y_rgb_1[-1])
        fig_y_rgb_1.append(fig_y_rgb[i])
        fig_y_opf_1.append(fig_y_opf_1[-1])
        fig_y_opf_1.append(fig_y_opf[i])
    fig_x_1.append(duration)
    fig_y_1.append(fig_y[-1])
    fig_y_rgb_1.append(fig_y_rgb[-1])
    fig_y_opf_1.append(fig_y_opf[-1])
    # plt.figure()
    # plt.plot(fig_x_1, fig_y_1, linewidth=2, color='lightskyblue')
    # plt.xlabel('time/s')
    # plt.ylabel('Slipping probability')
    # plt.ylim(0, 1)
    # plt.xlim(0, duration + 0.2)
    # plt.title('two stream network')
    # plt.savefig('bucket1/' + video_title + 'timeLable.jpg')
    # plt.show()

    # plt.figure()
    # plt.plot(fig_x_1, fig_y_rgb_1, linewidth=2, color='skyblue')
    # plt.xlabel('time/s')
    # plt.ylabel('Slipping probability')
    # plt.ylim(0, 1)
    # plt.xlim(0, duration + 0.2)
    # plt.title('spatial stream network')
    # plt.savefig('bucket1/' + video_title + '_rgb_' + 'timeLable.jpg')
    # plt.show()

    plt.figure()
    plt.plot(fig_x_1, fig_y_opf_1, linewidth=2, color='blue')
    plt.xlabel('time/s')
    plt.ylabel('Slipping probability')
    plt.ylim(0, 1)
    plt.xlim(0, duration + 0.2)
    plt.title('motion stream network')
    plt.savefig(video_title + '_opt_' + 'timeLable.jpg')
    plt.show()
