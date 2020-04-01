import cv2
import os
# import numpy as np
from skimage.measure import compare_ssim


def remove_simillar_image_by_ssim(path):
    img_list = os.listdir(path)
    img_list.sort()
    save_list = []
    count_num = 0
    for i in range(len(img_list)):
        try:
            img = cv2.imread(os.path.join(path, img_list[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256, 256))
            count_num += 1
        except:
            continue
        if count_num == 1:
            save_list.append(img_list[i])
            continue
        # elif len(save_list) < 5:
        #     flag = True
        #     for j in range(len(save_list)):
        #         com_img = cv2.imread(os.path.join(path, save_list[j]))
        #         com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)
        #         com_img = cv2.resize(com_img, (256, 256))
        #         sim = compare_ssim(img, com_img)
        #         if sim > 0.8:
        #             os.remove(os.path.join(path,img_list[i]))
        #             flag = False
        #             break
        #     if flag:
        #         save_list.append(img_list[i])
        else:
            flag = True
            # for save_img in save_list[-5:]:
            save_img = save_list[-1]
            com_img = cv2.imread(os.path.join(path, save_img))
            com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)
            com_img = cv2.resize(com_img, (256, 256))
            sim = compare_ssim(img, com_img)
            if sim > 0.93:
                os.remove(os.path.join(path,img_list[i]))
                flag = False
                # break
            if flag:
                save_list.append(img_list[i])

if __name__=="__main__":
    data_path = 'video/video_data/video_data_497/'
    data_path_sub = os.listdir(data_path)
    for i, j in enumerate(data_path_sub):
        # video_path = 'video/v_Slipping_g05_c08/'
        video_path = data_path + j + '/'
        remove_simillar_image_by_ssim(video_path)
        video_path_sub = os.listdir(video_path)
        for i, j in enumerate(video_path_sub):
            os.renames(video_path + j, video_path + 'frame_' + str(i + 1).zfill(6) + '.jpg')
            img = cv2.imread(video_path + 'frame_' + str(i + 1).zfill(6) + '.jpg')
            if img.shape[0] > 450 and img.shape[1] > 450:
                img_re = cv2.resize(img, (0, 0), fx=256 / img.shape[0], fy=256 / img.shape[0])
                cv2.imwrite(video_path + 'frame_' + str(i + 1).zfill(6) + '.jpg', img_re)