import os

first_path = 'video_data_475/'
first_path_sub = os.listdir(first_path)

with open('UCF_list/testlist01.txt', 'w') as f:
    for i in range(len(first_path_sub)):
        if (i + 1) % 4 == 0:
            video_name = first_path_sub[i]
            video_class = first_path_sub[i].split('_')[1]
            f.write(video_class + '/' + video_name + '.mp4' + '\n')
    f.close()

with open('UCF_list/trainlist01.txt', 'w') as f:
    for i in range(len(first_path_sub)):
        if (i + 1) % 4 > 0:
            video_name = first_path_sub[i]
            video_class = first_path_sub[i].split('_')[1]
            class_num = 1 if video_class == 'Slipping' else 2
            f.write(video_class + '/' + video_name + '.mp4' + ' ' + str(class_num) +'\n')
    f.close()
