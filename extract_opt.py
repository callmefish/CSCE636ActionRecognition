import os
import numpy as np
import cv2
from glob import glob

def cal_for_frames(video_path, video_name, flow_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    
    frames.sort()
    # flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        # flow.append(tmp_flow)
        prev = curr
        if not os.path.exists(os.path.join(flow_path, video_name + '_u')):
            os.mkdir(os.path.join(flow_path, video_name + '_u'))
        cv2.imwrite(os.path.join(flow_path, video_name + '_u', "{:06d}.jpg".format(i+1)), tmp_flow[:, :, 0])
        if not os.path.exists(os.path.join(flow_path, video_name + '_v')):
            os.mkdir(os.path.join(flow_path, video_name + '_v'))
        cv2.imwrite(os.path.join(flow_path, video_name + '_v', "{:06d}.jpg".format(i+1)), tmp_flow[:, :, 1])

    return


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


# def save_flow(video_flows, video_name, flow_path):
#     for i, flow in enumerate(video_flows):
#         if not os.path.exists(os.path.join(flow_path, video_name + '_u')):
#             os.mkdir(os.path.join(flow_path, video_name + '_u'))
#         cv2.imwrite(os.path.join(flow_path, video_name + '_u', "{:06d}.jpg".format(i+1)), flow[:, :, 0])
#         if not os.path.exists(os.path.join(flow_path, video_name + '_v')):
#             os.mkdir(os.path.join(flow_path, video_name + '_v'))
#         cv2.imwrite(os.path.join(flow_path, video_name + '_v', "{:06d}.jpg".format(i+1)), flow[:, :, 1])


def extract_flow(video_path, video_name, flow_path):
    # flow = cal_for_frames(video_path, video_name, flow_path)
    cal_for_frames(video_path, video_name, flow_path)
    # save_flow(flow, video_name, flow_path)
    print('complete:' + flow_path + video_name)
    return


if __name__ == '__main__':
    data_path = 'video_data_497/'
    data_path_sub = os.listdir(data_path)
    data_path_sub.sort()
    # data_path_sub = ['v_NotSlipping_g02_c08', 'v_NotSlipping_g09_c07', 'v_NotSlipping_g15_c01', 'v_NotSlipping_g20_c06', 'v_NotSlipping_g20_c08']
    for i, j in enumerate(data_path_sub[3:]):
        video_path = data_path + j + '/'
        save_path = 'opt/'
        extract_flow(video_path, j, save_path)