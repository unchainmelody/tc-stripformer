from __future__ import print_function
import numpy as np
import torch
import cv2
import yaml
import os
from torch.autograd import Variable
from models.networks import get_generator
import torchvision
import time
import argparse
from skimage.io import imread

def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--weights_path', required=True, help='Weights path')
	return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    with open('config/config_Stripformer_gopro.yaml') as cfg:
        config = yaml.safe_load(cfg)
    blur_path = './datasets/GoPro/test/blur/'
    out_path = './out/Stripformer_GoPro_results'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model = get_generator(config['model'])
    model.load_state_dict(torch.load(args.weights_path))
    model = model.cuda()

    test_time = 0
    iteration = 0
    total_image_number = 1111

    # warm-up
    warm_up = 0
    print('Hardware warm-up')
    for file in os.listdir(blur_path):
        for img_name in os.listdir(blur_path + '/' + file):
            warm_up += 1
            img = imread(blur_path + '/' + file + '/' + img_name)
            img_tensor = torch.from_numpy((img).astype('float32')) - 0.5
            # print(img_tensor.shape)
            # img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            # img_tensor = torch.from_numpy(np.transpose(img).astype('float32')) - 0.5
            # img_tensor = torch.from_numpy(img.astype('float32')) - 0.5
            with torch.no_grad():
                img_tensor = Variable((img_tensor.unsqueeze(0)).unsqueeze(1)).cuda()
                # print(img_tensor.shape)
                result_image = model(img_tensor)#[5,1,256,256]
            if warm_up == 20:
                # print("*"*50)
                break
        break

    for file in os.listdir(blur_path):
        if not os.path.isdir(out_path + '/' + file):
            os.mkdir(out_path + '/' + file)
        for img_name in os.listdir(blur_path + '/' + file):
            img = imread(blur_path + '/' + file + '/' + img_name)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy((img).astype('float32')) - 0.5
            # img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            # img_tensor = torch.from_numpy(np.transpose(img).astype('float32')) - 0.5
            # img_tensor = torch.from_numpy(img.astype('float32')) - 0.5
            with torch.no_grad():
                iteration += 1
                img_tensor = Variable((img_tensor.unsqueeze(0)).unsqueeze(1)).cuda()

                start = time.time()
                result_image = model(img_tensor)
                stop = time.time()
                print('Image:{}/{}, CNN Runtime:{:.4f}'.format(iteration, total_image_number, (stop - start)))
                test_time += stop - start
                print('Average Runtime:{:.4f}'.format(test_time / float(iteration)))
                result_image = result_image + 0.5
                # result_image=torch.from_numpy(np.squeeze(result_image).cpu().numpy())
                # print(result_image)
                # print('###############################')
                result_image = result_image.cpu().numpy()
                # print(result_image)
                out_file_name = out_path + '/' + file + '/' + img_name
                # print(out_file_name)
                np.save(out_file_name, result_image)
                # torchvision.utils.save_image(result_image, out_file_name, nrow=1)