import cv2
import numpy as np
import os
import time
from tqdm import tqdm
import shutil
import argparse
from glob import glob
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from test_spatial_dataloader import *
from test_motion_dataloader import *
from utils import *
from network import *
import json
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='video test for two stream')
parser.add_argument('--batch-size', default=19, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--video_name', type=str, default='csce636test/6000_4.mp4')
arg = parser.parse_args()
rgb_whole_pred = {}
opf_whole_pred = {}


def main(start_frame):
    # spatial Prepare DataLoader
    spatial_data_loader = test_spatial_dataloader(
        BATCH_SIZE=arg.batch_size,
        num_workers=8,
        path='record/test/temp_chunk/',
    )
    spatial_test_loader = spatial_data_loader.run()
    spatial_model = MODEL(
        lr=arg.lr,
        batch_size=arg.batch_size,
        resume='555_rgb_model_best.pth.tar',
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
        path='record/test/temp_opf/'
    )
    motion_test_loader = motion_data_loader.run()
    motion_model = MODEL(
        test_loader=motion_test_loader,
        resume='475_opt_model_best.pth.tar',
        evaluate='evaluate',
        lr=arg.lr,
        batch_size=arg.batch_size,
        channel=10 * 2,
        start_frame=start_frame,
    )
    motion_model.run()


class MODEL():
    def __init__(self, lr, batch_size, resume, evaluate, test_loader, channel,
                 start_frame):
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.evaluate = evaluate
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.channel = channel
        self.start_frame = start_frame

    def build_model(self):
        self.model = resnet101(pretrained=True, channel=self.channel).cuda()

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume)
                self.model.load_state_dict(checkpoint['state_dict'])
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
        cv2.imwrite(os.path.join(flow_path, video_name + '_u', "{:06d}.jpg".format(i + 1)), tmp_flow[:, :, 0])
        if not os.path.exists(os.path.join(flow_path, video_name + '_v')):
            os.mkdir(os.path.join(flow_path, video_name + '_v'))
        cv2.imwrite(os.path.join(flow_path, video_name + '_v', "{:06d}.jpg".format(i + 1)), tmp_flow[:, :, 1])
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


def save_fig(x, y, title, save_path):
    plt.figure()
    plt.plot(x, y, linewidth=2, color='lightskyblue')
    plt.xlabel('time/s')
    plt.ylabel('Slipping probability')
    plt.ylim(0, 1.1)
    plt.xlim(0, duration + 0.2)
    plt.title(title)
    plt.savefig(save_path)
    # plt.show()

def revise_order(path, isRGB):
    path_sub = os.listdir(path)
    path_sub.sort()
    if isRGB:
        for j in range(len(path_sub)):
            old_name = path + path_sub[j]
            new_name = path + 'frame_' + str(j+1).zfill(6) + '.jpg'
            os.rename(old_name, new_name)
    else:
        for j in range(len(path_sub)):
            old_name = path + path_sub[j]
            new_name = path + str(j+1).zfill(6) + '.jpg'
            os.rename(old_name, new_name)

def make_sure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    make_sure_path('record/')
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
    make_sure_path(rgb_outPutDirName)
    make_sure_path(opf_outPutDirName)

    index = 0
    while True:
        res, image = cap.read()
        if not res:
            print('not res , not image')
            break
        else:
            if width < height:
                pad = int((height - width) // 2 + 1)
                image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
            image = cv2.resize(image, (342, 256))
            cv2.imwrite(rgb_outPutDirName + 'frame_' + str(index + 1).zfill(6) + '.jpg', image)
            index += 1
    print('extract rgb finished')
    extract_flow(rgb_outPutDirName, 'v_temp_opf', opf_outPutDirName)
    cap.release()
    time_lable = {}

    make_sure_path('record/test/')
    make_sure_path('record/test/temp_chunk/')
    make_sure_path('record/test/temp_opf/')
    make_sure_path('record/test/temp_opf/v_temp_opf_u/')
    make_sure_path('record/test/temp_opf/v_temp_opf_v/')
    frame = 1
    for i in range(1, 41):
        shutil.copyfile(rgb_outPutDirName + 'frame_' + "{:06d}.jpg".format(i),
                        'record/test/temp_chunk/' + 'frame_' + "{:06d}.jpg".format(i))
        shutil.copyfile(opf_outPutDirName + 'v_temp_opf_u/' + "{:06d}.jpg".format(i),
                        'record/test/temp_opf/v_temp_opf_u/' + "{:06d}.jpg".format(i))
        shutil.copyfile(opf_outPutDirName + 'v_temp_opf_v/' + "{:06d}.jpg".format(i),
                        'record/test/temp_opf/v_temp_opf_v/' + "{:06d}.jpg".format(i))
    frame = 41
    StartFrame = frame - 41
    main(StartFrame)

    while frame + 10 < index:
        for i in range(1, 11):
            os.remove('record/test/temp_chunk/' + 'frame_' + "{:06d}.jpg".format(i))
            shutil.copyfile(rgb_outPutDirName + 'frame_' + "{:06d}.jpg".format(frame),
                            'record/test/temp_chunk/' + 'frame_' + "{:06d}.jpg".format(frame))
            os.remove('record/test/temp_opf/v_temp_opf_u/' + "{:06d}.jpg".format(i))
            shutil.copyfile(opf_outPutDirName + 'v_temp_opf_u/' + "{:06d}.jpg".format(frame),
                            'record/test/temp_opf/v_temp_opf_u/' + "{:06d}.jpg".format(frame))
            os.remove('record/test/temp_opf/v_temp_opf_v/' + "{:06d}.jpg".format(i))
            shutil.copyfile(opf_outPutDirName + 'v_temp_opf_v/' + "{:06d}.jpg".format(frame),
                            'record/test/temp_opf/v_temp_opf_v/' + "{:06d}.jpg".format(frame))
            frame += 1
        revise_order('record/test/temp_chunk/', True)
        revise_order('record/test/temp_opf/v_temp_opf_u/', False)
        revise_order('record/test/temp_opf/v_temp_opf_v/', False)
        StartFrame += 10
        main(StartFrame)


    fig_x, fig_y, fig_y_rgb, fig_y_opf = [], [], [], []
    for key in list(rgb_whole_pred.keys()):
        cur_time = float(key) / rate
        new_key = str(float('%.3f' % cur_time))
        new_value = softmax(rgb_whole_pred[key] + 1 * opf_whole_pred[key]).tolist()
        rgb_value = softmax(rgb_whole_pred[key]).tolist()
        opf_value = softmax(opf_whole_pred[key]).tolist()
        time_lable[new_key] = new_value[0]
        fig_x.append(cur_time)
        fig_y.append(new_value[0])
        fig_y_rgb.append(rgb_value[0])
        fig_y_opf.append(opf_value[0])

    point_num = len(fig_y_rgb)
    one_count = 0
    zero_count = 0
    for i in fig_y_rgb:
        if abs(i-1) < 1e-4:
            one_count += 1
        if abs(i-0) < 1e-4:
            zero_count += 1
    if one_count/point_num > 0.9:
        time_lable = {str(float('%.3f'%(float(key)/rate))): softmax(opf_whole_pred[key]).tolist()[0] for key in
                      list(rgb_whole_pred.keys())}
        fig_y = fig_y_opf.copy()
    elif zero_count / point_num > 0.9:
        time_lable = {str(float('%.3f' % (float(key) / rate))): softmax(rgb_whole_pred[key]).tolist()[0] for key in
                      list(rgb_whole_pred.keys())}
        fig_y = fig_y_rgb.copy()
    
    json_str = json.dumps(time_label)
    with open(video_title + '_' + 'timelabel.json', 'w') as json_file:
        json_file.write(json_str)
        json_file.close()

    fig_x_1 = fig_x[:1]
    fig_y_1 = fig_y[:1]
    fig_y_rgb_1 = fig_y_rgb[:1]
    fig_y_opf_1 = fig_y_opf[:1]

    for i in range(1, len(fig_x)):
        fig_x_1.append(fig_x[i] - 0.001)
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

    save_fig(fig_x_1, fig_y_1, 'two stream network', 'bucket3/' + video_title + '_Part6.jpg')
#     save_fig(fig_x_1, fig_y_rgb_1, 'spatial stream network', 'bucket3/' + video_title + '_rgb' + '_Part6.jpg')
#     save_fig(fig_x_1, fig_y_opf_1, 'motion stream network', 'bucket3/' + video_title + '_opf' + '_Part6.jpg')
    rgb_whole_pred = {}
    opf_whole_pred = {}
    shutil.rmtree('record/')
