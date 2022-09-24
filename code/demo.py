# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot,SiamRPNBIG,SiamRPNotb
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net = SiamRPNotb()
net.load_state_dict(torch.load('SiamRPNOTB.model'))
net.eval().cuda()

# # image and init box
# image_files = sorted(glob.glob('./bag/*.jpg'))
# init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]

path_gt = "/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/anno/UAV123/car1_s.txt" 
img_files_path = glob.glob("/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/*")
img_files_path.sort()

my_file = open(path_gt)
line = my_file.readline()
init_rbox = [int(l) for l in line[:-1].split(',')]
my_file.close()


# [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
cx = init_rbox[0] + init_rbox[2]/2
cy = init_rbox[1] + init_rbox[3]/2
w = init_rbox[2]
h = init_rbox[3]
# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(img_files_path[0])  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net)

# tracking and visualization
toc = 0
for f, image_file in enumerate(img_files_path):
    im = cv2.imread(image_file)
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, im)  # track
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    cv2.imshow('SiamRPN', im)
    cv2.waitKey(1)

print('Tracking Speed {:.1f}fps'.format((len(img_files_path)-1)/(toc/cv2.getTickFrequency())))
