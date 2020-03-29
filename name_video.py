import os

first_path = 'video/new_video/'
first_path_sub = os.listdir(first_path)

for i in first_path_sub:
    second_path = first_path + i + '/'
    second_path_sub = os.listdir(second_path)
    os.renames(second_path, first_path + i + 'tem' + '/')

first_path_sub_new = os.listdir(first_path)
num_g = 21
num_c = 1

for i in first_path_sub_new:
    second_path = first_path + i + '/'
    second_path_sub = os.listdir(second_path)
    index = 1
    for j in second_path_sub:
        os.renames(second_path + j, second_path + 'frame_' + str(index).zfill(6) + '.jpg')
        index += 1
    video_name = 'v_NotSlipping' + '_g' + str(num_g).zfill(2) + '_c' + str(num_c).zfill(2)
    os.renames(second_path, first_path + video_name + '/')
    num_c += 1
    if num_c > 10:
        num_g += 1
        num_c = 1