# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from FreqSal import FreqSal
from options import opt
from data import test_dataset
import time

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./test_datasets/',help='test dataset path')
opt = parser.parse_args()

torch.cuda.empty_cache()

dataset_path = opt.test_path

#load the model
model = FreqSal()

gpu_num = torch.cuda.device_count()
# load gpu
if gpu_num == 1:
    print("Use Single GPU-", opt.gpu_id)
elif gpu_num > 1:
    print("Use multiple GPUs-", opt.gpu_id)
    model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load('./cpts/FreqSal_RGBT_best.pth'))

model.cuda()
model.eval()

# test
# test_datasets = ['SIP','NJU2K','NLPR','STERE','DES','DUT-RGBD']
test_datasets = ['VT821','VT1000','VT5000','VI-RGBT1500']
t_all = []
for dataset in test_datasets:
    save_path = './test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth = depth.repeat(1,3,1,1).cuda()
        # depth = depth.cuda()
        time1 = time.time()
        with torch.no_grad():
            res = model(image, depth)
        time2 = time.time()
        t_all.append(time2 - time1)
        res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res*255)
    print('Test Done!')
    print('average time:{:.02f} s'.format(np.mean(t_all) / 1))
    print('average FPS :{:.02f} fps'.format(1 / np.mean(t_all)))

