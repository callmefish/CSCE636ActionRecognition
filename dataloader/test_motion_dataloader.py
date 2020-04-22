import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
# from .split_train_test_video import *
from skimage import io, color, exposure
import torch
import os


class test_motion_dataset(Dataset):
    def __init__(self, dic, in_channel, root_dir, transform=None):

        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.transform = transform
        self.in_channel = in_channel
        self.img_rows = 224
        self.img_cols = 224

    def stackopf(self):
        name = 'v_' + self.video
        u = self.root_dir + name + '_u'
        v = self.root_dir + name + '_v'

        # built float tensor, which torch.size is [20, 224, 224]
        flow = torch.FloatTensor(2 * self.in_channel, self.img_rows, self.img_cols)
        i = int(self.clips_idx)

        for j in range(self.in_channel):
            idx = i + j
            idx = str(idx)
            frame_idx = idx.zfill(6)
            h_image = u + '/' + frame_idx + '.jpg'
            v_image = v + '/' + frame_idx + '.jpg'

            imgH = (Image.open(h_image))
            imgV = (Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            flow[2 * (j - 1), :, :] = H
            flow[2 * (j - 1) + 1, :, :] = V
            imgH.close()
            imgV.close()
        return flow
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)

        self.video, self.clips_idx = self.keys[idx].split('-')

        label = self.values[idx]
        label = int(label) - 1
        data = self.stackopf()

        sample = (self.video, data, label)

        return sample

class test_motion_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel, path):

        self.BATCH_SIZE = BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count = {}
        self.in_channel = in_channel
        self.data_path = path
        self.test_video = {'temp_opf': 2}


    def load_frame_count(self):
        videoname = 'temp_opf'
        self.frame_count[videoname] = len(os.listdir(self.data_path + '/v_temp_opf_u/'))


    def run(self):
        self.load_frame_count()
        self.val_sample19()
        val_loader = self.val()

        return val_loader

    def val_sample19(self):
        self.dic_test_idx = {}
        for video in self.test_video:
            sampling_interval = int((self.frame_count[video] - 10 + 1) / 19)
            for index in range(19):
                clip_idx = index * sampling_interval
                key = video + '-' + str(clip_idx + 1)
                self.dic_test_idx[key] = self.test_video[video]

    def val(self):
        
        validation_set = test_motion_dataset(dic=self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path,
                                        transform=transforms.Compose([
                                            transforms.Resize([224, 224]),
                                            transforms.ToTensor(),
                                        ]))


        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader


if __name__ == '__main__':
    dataloader = test_motion_dataloader(BATCH_SIZE=1, num_workers=1, in_channel=20,
                                         path='/home/yzy20161103/csce636project/project/record/temp_opf/')
    val_loader = dataloader.run()
