import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
# from .split_train_test_video import *
from skimage import io, color, exposure
import torch
import os

class test_spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = list(dic.keys())
        self.values = list(dic.values())
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self,video_name, index):

        path = self.root_dir + 'frame_'
        img = Image.open(path + str(index).zfill(6)+'.jpg')
        transformed_img = self.transform(img)
        img.close()
        return transformed_img

    # 让对象实现迭代功能
    def __getitem__(self, idx):
        if self.mode == 'val':
            video_name, index = self.keys[idx-1].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only val mode')

        label = self.values[idx-1]
        label = int(label)-1
        
        if self.mode == 'val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only val mode')
           
        return sample

class test_spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count = {}
        self.test_video = {'temp_chunk': 2}

    # 把video名称和帧数对应起来的字典, {'asdasd': 289, 'asdasc': 152}
    def load_frame_count(self):
        videoname = 'temp_chunk'
        self.frame_count[videoname] = len(os.listdir(self.data_path))

    def run(self):
        self.load_frame_count()
        self.val_sample20()
        val_loader = self.validate()

        return val_loader
                    
    def val_sample20(self):
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            # 分成19个片段
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video + ' ' + str(frame+1)
                self.dic_testing[key] = self.test_video[video]      

    def validate(self):
        validation_set = test_spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        #print('==> Validation data :',len(validation_set),'frames')
        # sample = (video_name, data, label), extract data
        #print(validation_set[1][1].size())

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader





if __name__ == '__main__':
    
    dataloader = test_spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                path='/home/yzy20161103/csce636project/two-stream-action-recognition/record/temp_chunk/')
    val_loader = dataloader.run()
