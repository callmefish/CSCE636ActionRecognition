'''
write down the shape of video data frame
'''


import os
import cv2


first_dir = 'video/video_data/video_data_497/'
first_dir_sub = os.listdir(first_dir)
shape_list = []
for i in first_dir_sub:
    second_dir = first_dir + i + '/'
    second_dir_sub = os.listdir(second_dir)
    img_path = second_dir + second_dir_sub[0]
    img = cv2.imread(img_path)
    size = img.shape
    img_shape = [img.shape[0], img.shape[1]]
    if img_shape not in shape_list:
        shape_list.append(img_shape)
    # if img_shape == [720, 406]:
    #     print(second_dir)

print(shape_list)