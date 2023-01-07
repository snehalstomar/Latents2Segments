import options
import data
import models
from evaluation import GroupEvaluator
import os 
import numpy as  np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_init import get_opt
from options import TestOptions
import data
import models
from evaluation import GroupEvaluator
import os
import torchvision.transforms as transforms
from PIL import Image
from evaluation import BaseEvaluator
from data.base_dataset import get_transform
import util
import cv2
import os
import time

opt = TestOptions().parse()
print('opt-->', opt)
dataset = data.create_dataset(opt)
evaluators = GroupEvaluator(opt)
model = models.create_model(opt)

def load_image(path):
    path = os.path.expanduser(path)
    img = Image.open(path).convert('RGB')
    transform = get_transform(opt)
    tensor = transform(img).unsqueeze(0)
    return tensor

imgdir = 'swapping_style_controlled_AE_dataset/test/full'
output_dir = 'parsing_results/metric_pour_paper_1'
img_num = 8613
img_path = imgdir + '/' + str(img_num) + '.png'  
target_dir = output_dir + '/' + str(img_num)
os.mkdir(target_dir)
#img2_path = 'testphotos/ffhq512/137.png'

img = load_image(img_path)
#img2 = load_image(img2_path)
model(sample_image=img, command="fix_noise")
curr_time_1 = time.time()
S_s, S_t = model(img, command="encode")
print(S_s.shape, S_t.shape)

S_s_zeros = torch.zeros(1, 8, 8, 8)
S_t_zeros = torch.zeros(1, 2048)
struct_slice_ones = torch.ones(1, 2, 8, 8)
struct_slice_ones_1 = torch.ones(1, 1, 8, 8)

struct_mask_hair, struct_mask_skin, struct_mask_nose, struct_mask_eyes, struct_mask_lips = torch.zeros(1, 8, 8, 8), torch.zeros(1, 8, 8, 8), torch.zeros(1, 8, 8, 8), torch.zeros(1, 8, 8, 8), torch.zeros(1, 8, 8, 8) 
struct_mask_hair[0, 0: 2, :, :] = struct_slice_ones
struct_mask_skin[0, 2: 4, :, :] = struct_slice_ones
struct_mask_nose[0, 4: 6, :, :] = struct_slice_ones
struct_mask_eyes[0, 6: 7, :, :] = struct_slice_ones_1
struct_mask_lips[0, 7: 8, :, :] = struct_slice_ones_1

S_s_hair = torch.mul(S_s, struct_mask_hair.to('cuda'))
S_s_skin = torch.mul(S_s, struct_mask_skin.to('cuda')) 
S_s_nose = torch.mul(S_s, struct_mask_nose.to('cuda'))
S_s_eyes = torch.mul(S_s, struct_mask_eyes.to('cuda'))
S_s_lips = torch.mul(S_s, struct_mask_lips.to('cuda'))

output_image_as_it_is = model(S_s, S_t, command="decode")
output_image_as_it_is = transforms.ToPILImage()((output_image_as_it_is[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
output_image_as_it_is.save(target_dir + '/full_recon.png')


output_image_hair = model(S_s_hair, S_t, command="decode")
output_image_hair = transforms.ToPILImage()((output_image_hair[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
output_image_hair.save(target_dir + '/hair_recon.png')

output_image_skin = model(S_s_skin, S_t, command="decode")
output_image_skin = transforms.ToPILImage()((output_image_skin[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
output_image_skin.save(target_dir + '/skin_recon.png')

output_image_nose = model(S_s_nose, S_t, command="decode")
output_image_nose = transforms.ToPILImage()((output_image_nose[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
output_image_nose.save(target_dir + '/nose_recon.png')

output_image_eyes = model(S_s_eyes, S_t, command="decode")
output_image_eyes = transforms.ToPILImage()((output_image_eyes[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
output_image_eyes.save(target_dir + '/eyes_recon.png')

output_image_lips = model(S_s_lips, S_t, command="decode")
output_image_lips = transforms.ToPILImage()((output_image_lips[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
output_image_lips.save(target_dir + '/lips_recon.png')

hair_img = cv2.imread(target_dir + '/hair_recon.png')
skin_img = cv2.imread(target_dir + '/skin_recon.png')
nose_img = cv2.imread(target_dir + '/nose_recon.png')
eyes_img = cv2.imread(target_dir + '/eyes_recon.png')
lips_img = cv2.imread(target_dir + '/lips_recon.png')

hair_img = np.sum(hair_img, 2) / 3
skin_img = np.sum(skin_img, 2) / 3
nose_img = np.sum(nose_img, 2) / 3
eyes_img = np.sum(eyes_img, 2) / 3
lips_img = np.sum(lips_img, 2) / 3

parsed_image = np.zeros([128, 128, 3])

hair_nz_x_y = np.where(hair_img != 0)
skin_nz_x_y = np.where(skin_img != 0)
nose_nz_x_y = np.where(nose_img != 0)
eyes_nz_x_y = np.where(eyes_img != 0)
lips_nz_x_y = np.where(lips_img != 0)

for x,y in zip(hair_nz_x_y[0], hair_nz_x_y[1]):
    parsed_image[x, y, 0], parsed_image[x, y, 1], parsed_image[x, y, 2] = 255, 0, 0
for x,y in zip(skin_nz_x_y[0], skin_nz_x_y[1]):
    parsed_image[x, y, 0], parsed_image[x, y, 1], parsed_image[x, y, 2] = 0, 255, 0
for x,y in zip(nose_nz_x_y[0], nose_nz_x_y[1]):
    parsed_image[x, y, 0], parsed_image[x, y, 1], parsed_image[x, y, 2] = 0, 0, 255
for x,y in zip(eyes_nz_x_y[0], eyes_nz_x_y[1]):
    parsed_image[x, y, 0], parsed_image[x, y, 1], parsed_image[x, y, 2] = 0, 128, 255
for x,y in zip(lips_nz_x_y[0], lips_nz_x_y[1]):
    parsed_image[x, y, 0], parsed_image[x, y, 1], parsed_image[x, y, 2] = 70, 70, 70
curr_time_2 = time.time()

print('time_taken in s->', curr_time_2 - curr_time_1)
# cv2.imwrite(target_dir + '/parsed_image.png', parsed_image)
# cv2.imwrite(target_dir + '/input_image.png', cv2.imread(img_path))