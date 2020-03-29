import os

first_path = 'video/video_data/video_data/'
first_path_sub = os.listdir(first_path)

with open('testlist01.txt', 'w') as f:
    for i in range(len(first_path_sub)):
        if (i + 1) % 4 == 0:
            video_name = first_path_sub[i]
            video_class = first_path_sub[i].split('_')[1]
            f.write(video_class + '/' + video_class + '\n')
    f.close()

with open('trainlist01.txt', 'w') as f:
    for i in range(len(first_path_sub)):
        if (i + 1) % 4 > 0:
            video_name = first_path_sub[i]
            video_class = first_path_sub[i].split('_')[1]
            class_num = 1 if video_class == 'Slipping' else 2
            f.write(video_class + '/' + video_class + ' ' + '\n')
    f.close()
