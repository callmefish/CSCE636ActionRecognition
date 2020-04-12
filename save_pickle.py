import pickle
import os

first_dir = 'video_data_475/'
first_dir_sub = os.listdir(first_dir)

frame_count = {}

for i in first_dir_sub:
    number = len(os.listdir(first_dir + i))
    frame_count[i] = number


pickle_file = open('dataloader/dic/frame_count.pickle', 'wb')
pickle.dump(frame_count, pickle_file)
pickle_file.close()
