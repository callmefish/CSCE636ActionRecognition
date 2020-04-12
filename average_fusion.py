from utils import *
import dataloader

if __name__ == '__main__':

    rgb_preds='record/spatial_497_5/spatial_video_preds.pickle'
    opf_preds = 'record/motion/motion_video_preds.pickle'

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1,
                                    path='/home/yzy20161103/csce636_project/project/video_data_497/',
                                    ucf_list='/home/yzy20161103/csce636_project/project/UCF_list/',
                                    ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros((len(rgb.keys()),2))
    video_level_labels = np.zeros(len(rgb.keys()))
    correct=0
    ii=0
    a = list(rgb.keys())
    a.sort()
    for name in a:
        
        r = rgb[name]
#         if name in opf.keys():
#             o = opf[name]
#         else:
#             o = [0, 0]
        o = [0,0]    

        label = int(test_video[name])-1

        video_level_preds[ii,:] = (r+o)
        video_level_labels[ii] = label
        ii+=1
        if np.argmax(r+o) == (label):
            correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()

    top1 = accuracy(video_level_preds, video_level_labels, topk=(1,))

    print(top1)
