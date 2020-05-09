#!/bin/bash
#replace the variables with your github repo url, repo name, test
#video name, json named by your UIN
GIT_REPO_URL="https://github.com/callmefish/CSCE636ActionRecognition.git"
REPO="CSCE636ActionRecognition"
VIDEO="./csce636test/5900_3.mp4"
UIN_JSON="529005218.json"
UIN_JPG="529005218.jpg"
git clone $GIT_REPO_URL
cd $REPO
gsutil cp gs://ucf101_for_rar/475_opt_model_best.pth.tar ./
gsutil cp gs://ucf101_for_rar/555_rgb_model_best.pth.tar ./
#Replace this line with commands for running your test python file.
echo $VIDEO
python test.py --video_name $VIDEO
python test.py --video_name "./csce636test/6000_4.mp4"
python test.py --video_name "./csce636test/6100_2.mp4"
python test.py --video_name "./csce636test/6200_1.mp4"
python test.py --video_name "./csce636test/6300_1.mp4"
python test.py --video_name "./csce636test/6300_2.mp4"
python test.py --video_name "./csce636test/6300_3.mp4"
python test.py --video_name "./csce636test/6300_4.mp4"
python test.py --video_name "./csce636test/6400_3.mp4"
python test.py --video_name "./csce636test/6500_1.mp4"
python test.py --video_name "./csce636test/6600_4.mp4"
python test.py --video_name "./csce636test/6700_2.mp4"