import os, pickle


class UCF101_splitter():
    def __init__(self, path, split):
        self.path = path
        self.split = split

    # 建立一个动作编号和动作对应的字典, {'ball': 1, 'swim': 2}
    def get_action_index(self):
        self.action_label={}
        with open(self.path+'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label,action = line.split(' ')
            #print label,action
            if action not in self.action_label.keys():
                self.action_label[action]=label


    def split_video(self):
        self.get_action_index()
        # 逐一探索子文件夹底下的文件
        for path,subdir,files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    train_video = self.file2_dic(self.path+filename)
                if filename.split('.')[0] == 'testlist'+self.split:
                    test_video = self.file2_dic(self.path+filename)
        print('==> (Training video, Validation video):(', len(train_video),len(test_video),')')
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)

        return self.train_video, self.test_video

    # 将video名字和前面定好的动作编号对应在一个字典里, {'asdasd': 1, 'asdasdasda': 2}
    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic={}
        for line in content:
            #print line
            video = line.split('/',1)[1].split(' ',1)[0]
            key = video.split('_',1)[1].split('.',1)[0]
            label = self.action_label[line.split('/')[0]]   
            dic[key] = int(label)
            #print key,label
        return dic

    # 用处就是把大写的S改成小写的s，其他照常
    def name_HandstandPushups(self,dic):
        dic2 = {}
        for video in dic:
            # YoYo_g25_c05  ==> n = 'YoYo', g = 'g25_c05'
            n,g = video.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            else:
                videoname=video
            dic2[videoname] = dic[video]
        return dic2


if __name__ == '__main__':

    path = '/home/yzy20161103/two-stream-action-recognition/UCF_list/'
    split = '01'
    splitter = UCF101_splitter(path=path,split=split)
    train_video,test_video = splitter.split_video()
    print(len(train_video),len(test_video))
