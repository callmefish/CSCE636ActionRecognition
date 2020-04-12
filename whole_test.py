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

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import test_spatial_dataloader

from utils import *
from network import *
import math
import json
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# 添加命令行指令
parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=0, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', default='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='/home/yzy20161103/csce636_project/project/record/spatial_497_5/model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main(start_frame):
    global arg
    global whole_dic
    arg = parser.parse_args()

    #Prepare DataLoader
    data_loader = test_spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        # 进程数量
                        num_workers=8,
                        path='/home/yzy20161103/csce636_project/project/record/temp_chunk/',
                        )
    
    test_loader = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        test_loader=test_loader,
                        start_frame=start_frame,
    )
    #Training
    model.run()

class Spatial_CNN():
    global whole_dic
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, test_loader,start_frame):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.test_loader=test_loader
        self.best_prec1=0
        self.start_frame=start_frame

    def build_model(self):
        #build model
        self.model = resnet101(pretrained= True, channel=3).cuda()
        #self.model = nn.DataParallel(resnet101(pretrained=True, channel=3)).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
#                 print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
#                   .format(self.resume, checkpoint['epoch'], self.best_prec1))
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
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        #progress = self.test_loader
        with torch.no_grad():
            for i, (keys,data,label) in enumerate(progress):
                data = data.cuda()

                # compute output
                output = self.model(data)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                #Calculate video level prediction
                preds = output.data.cpu().numpy()
                nb_data = preds.shape[0]
                for j in range(nb_data):
                    videoName = keys[j].split('/',1)[0] + '_' + str(self.start_frame)
                    if videoName not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[videoName] = preds[j,:]
                        whole_dic[videoName] = preds[j,:]
                    else:
                        self.dic_video_level_preds[videoName] += preds[j,:]
                        whole_dic[videoName] += preds[j,:]

        return 


    

if __name__=='__main__':
    file_path = '/home/yzy20161103/csce636_project/project/record/sample_video/sample_video_05(N).mp4'
    video_title = file_path.split('/')[-1][:-4]
    print(video_title)
    cap = cv2.VideoCapture(file_path)    
    # file_path是文件的绝对路径，防止路径中含有中文时报错，需要解码
    if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
        # get方法参数按顺序对应下表（从0开始编号)
        rate = cap.get(5)   # 帧速率
        FrameNumber = cap.get(7)  # 视频文件的帧数
        duration = FrameNumber/rate  # 帧速率/视频总帧数 是时间，秒
        print(duration)
    outPutDirName = '/home/yzy20161103/csce636_project/project/record/temp_chunk/'

    if not os.path.exists(outPutDirName):
        # 如果文件目录不存在则创建目录
        os.makedirs(outPutDirName)

    frame = 1
    index = 0
    whole_dic = {}

    while True:
        res, image = cap.read()
        if not res:
            print('not res , not image')
            if len(os.listdir(outPutDirName)) > 0:
                StartFrame = frame - len(os.listdir(outPutDirName))
                main(StartFrame)
                shutil.rmtree(outPutDirName)
            break
        else:
            cv2.imwrite(outPutDirName + 'frame_' + str(index+1).zfill(6) + '.jpg', image)
            frame += 1
            index += 1
            if (frame - 1)%100 == 0:
                StartFrame = frame - 101
                main(StartFrame)
                #break
                shutil.rmtree(outPutDirName)
                os.makedirs(outPutDirName)
                index = 0
        
    #print('whole_dic : {}'.format(whole_dic))
    print('finish video analysis')
    cap.release()
    time_label = {'Slipping': []}
    list_key = list(whole_dic.keys())
    x = []
    y = []
    for i in range(len(list_key)):
        current_time = int(list_key[i].split('_')[2])/rate
        x.append(current_time)
        #softmax
        value = whole_dic[list_key[i]]
        prec = math.exp(value[0]) / (math.exp(value[0]) + math.exp(value[1]))
        y.append(prec)
        time_label['Slipping'].append([current_time, prec])
    print('time_label : {}'.format(time_label))
    #os.makedirs('/home/yzy20161103/two-stream-action-recognition/record/time_label_data.json')
    json_str = json.dumps(time_label)
    with open('/home/yzy20161103/csce636_project/project/time_label_data_' + video_title + '.json', 'w') as json_file:
        json_file.write(json_str)
    plt.plot(x, y, linewidth=3, color='blue')
    plt.xlabel('time/s')
    plt.ylabel('Slipping')
    plt.savefig('/home/yzy20161103/csce636_project/project/result_' + video_title + '.png')
    plt.show()
