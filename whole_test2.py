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
parser.add_argument('--batch-size', default=19, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')


def main(start_frame):
    global arg
    global whole_dic
    arg = parser.parse_args()

    # spatial Prepare DataLoader
    spatial_data_loader = test_spatial_dataloader(
        BATCH_SIZE=arg.batch_size,
        num_workers=8,
        path='/home/yzy20161103/demo/CSCE636ActionRecognition/record/temp_chunk/',
    )
    spatial_test_loader = spatial_data_loader.run()
    # spatial Model
    spatial_model = Spatial_CNN(
        lr=arg.lr,
        batch_size=arg.batch_size,
        resume='/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/spatial/model_best.pth.tar',
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
        path='/home/yzy20161103/demo/CSCE636ActionRecognition/record/temp_opf/'
    )
    motion_test_loader = motion_data_loader.run()
    # motion Model
    motion_model = Motion_CNN(
        test_loader=motion_test_loader,
        start_epoch=0,
        resume='/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/motion/model_best.pth.tar',
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


if __name__ == '__main__':
    start_time = time.time()
    file_path = '/home/yzy20161103/demo/CSCE636ActionRecognition/record/sample_video/sample_video_04(S).mp4'
    video_title = file_path.split('/')[-1][:-4]
    print(video_title)
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():  
        rate = cap.get(5)  
        FrameNumber = cap.get(7)  
        duration = FrameNumber / rate  
        print(duration)
    rgb_outPutDirName = '/home/yzy20161103/demo/CSCE636ActionRecognition/record/temp_chunk/'
    opf_outPutDirName = '/home/yzy20161103/demo/CSCE636ActionRecognition/record/temp_opf/'

    if not os.path.exists(rgb_outPutDirName):
        os.makedirs(rgb_outPutDirName)
    if not os.path.exists(opf_outPutDirName):
        os.makedirs(opf_outPutDirName)

    frame = 1
    index = 0
    whole_dic_spatial = {}
    whole_dic_motion = {}

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
    time_label = {'Slipping': []}
    whole_dic = {}
    for key, value in whole_dic_motion.items():
        whole_dic_motion[key] = value.tolist()
    whole_dic_spatial = {k: v.tolist() for k, v in whole_dic_spatial.items()}
    with open('/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/' + video_title + '_motion.json', 'w') as json_file:
        json_str0 = json.dumps(whole_dic_motion)
        json_file.write(json_str0)
        json_file.close()
    with open('/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/' + video_title + '_spatial.json', 'w') as json_file:
        json_str1 = json.dumps(whole_dic_spatial)
        json_file.write(json_str1)
        json_file.close()
    
    
    for key in whole_dic_spatial.keys():
        new_key = 'temp_opf_' + key.split('_')[2]
        whole_dic[key] = [i+j for i,j in zip(whole_dic_spatial[key],whole_dic_motion[new_key])]
    with open('/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/' + video_title + '_sum.json', 'w') as json_file:
        json_str2 = json.dumps(whole_dic)
        json_file.write(json_str2)
        json_file.close()

    list_key_sum = list(whole_dic.keys())
    list_key_rgb = list(whole_dic_spatial.keys())
    list_key_opt = list(whole_dic_motion.keys())
    x = []
    y_sum = []
    y_rgb = []
    y_opt = []
    for i in range(len(list_key_sum)):
        current_time = int(list_key_sum[i].split('_')[2]) / rate
        x.append(current_time)
        # softmax
        value = whole_dic[list_key_sum[i]]
        prec_sum = math.exp(value[0]) / (math.exp(value[0]) + math.exp(value[1]))
        y_sum.append(prec_sum)
        time_label['Slipping'].append([current_time, prec_sum])
        value = whole_dic_spatial[list_key_rgb[i]]
        prec_rgb = math.exp(value[0]) / (math.exp(value[0]) + math.exp(value[1]))
        y_rgb.append(prec_rgb)
        value = whole_dic_motion[list_key_opt[i]]
        prec_opt = math.exp(value[0]) / (math.exp(value[0]) + math.exp(value[1]))
        y_opt.append(prec_opt)
    print('time_label : {}'.format(time_label))
    end_time = time.time()
    print('The running time of video test is {}'.format(end_time - start_time))
    with open('/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/time_label_data_' + video_title + '.json', 'w') as json_file:
        json_str = json.dumps(time_label)
        json_file.write(json_str)
        json_file.close()
    plt.figure()
    plt.plot(x, y_sum, linewidth=2, color='lightskyblue')
    plt.xlabel('time/s')
    plt.ylabel('Slipping')
    plt.title('average fusion result')
    plt.savefig('/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/result_' + video_title + '_sum.png')
    plt.show()
    
    plt.figure()
    plt.plot(x, y_rgb, linewidth=2, color='lightskyblue')
    plt.xlabel('time/s')
    plt.ylabel('Slipping')
    plt.title('spatial stream result')
    plt.savefig('/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/result_' + video_title + '_rgb.png')
    plt.show()
    
    plt.figure()
    plt.plot(x, y_opt, linewidth=2, color='lightskyblue')
    plt.xlabel('time/s')
    plt.ylabel('Slipping')
    plt.title('motion stream result')
    plt.savefig('/home/yzy20161103/demo/CSCE636ActionRecognition/best_model_475/result_' + video_title + '_opt.png')
    plt.show()
