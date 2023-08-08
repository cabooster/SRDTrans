import imp
from operator import imod
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import os
import math
import tifffile as tiff
import numpy as np
import random

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')")

parser.add_argument('--patch_x', type=int, default=128, help="patch size in x and y")
parser.add_argument('--patch_t', type=int, default=128, help="patch size in t")
parser.add_argument('--overlap_factor', type=float, default=0.5, help="the overlap factor between two adjacent patches")

parser.add_argument('--train_datasets_size', type=int, default=6000, help='How many patches will be used for training.')
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")


parser.add_argument('--pth_path', type=str, default='./pth', help="the root path to save models")
parser.add_argument('--datasets_folder', type=str, default='./train', help="A folder containing files for training")
parser.add_argument('--output_path', type=str, default='./results', help="output directory")


parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")

parser.add_argument('--select_img_num', type=int, default=10000000000, help='How many frames will be used for training.')
parser.add_argument('--test_datasize', type=int, default=10000000000, help='How many frames will be tested.')
parser.add_argument('--scale_factor', type=int, default=1, help='the factor for image intensity scaling')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU

#############################################################################################################################################
from SRDTrans import SRDTrans
from data_process import train_preprocess_lessMemoryMulStacks, trainset
from utils import save_yaml_train
from sampling import *

########################################################################################################################
# use isotropic patch size by default
opt.patch_y = opt.patch_x   # the height of 3D patches (patch size in y)
opt.patch_t = opt.patch_t   # the length of 3D patches (patch size in t)
opt.gap_x = int(opt.patch_x*(1-opt.overlap_factor))     # patch gap in x
opt.gap_y = int(opt.patch_y*(1-opt.overlap_factor))     # patch gap in y
opt.gap_t = int(opt.patch_t*(1-opt.overlap_factor))     # patch gap in t
opt.ngpu = opt.GPU.count(',')+1
opt.batch_size = opt.ngpu                               # By default, the batch size is equal to the number of GPU for minimal memory consumption
print('\033[1;31mTraining parameters -----> \033[0m')
print(opt)

########################################################################################################################
if not os.path.exists(opt.output_path): 
    os.mkdir(opt.output_path)
current_time = opt.datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
output_path = opt.output_path + '/' + current_time
pth_path = 'pth//'+ current_time
print("ckp is saved in {}".format(pth_path))
if not os.path.exists(pth_path): 
    os.mkdir(pth_path)

train_name_list, train_noise_img, train_coordinate_list, stack_index = train_preprocess_lessMemoryMulStacks(opt)

yaml_name = pth_path+'//para.yaml'
save_yaml_train(opt, yaml_name)
########################################################################################################################

L1_pixelwise = torch.nn.L1Loss()
L2_pixelwise = torch.nn.MSELoss()

denoise_generator = SRDTrans(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128*4,
    window_size=7,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

param_num = sum([param.nelement() for param in denoise_generator.parameters()])
print('\033[1;31mParameters of the model is {:.2f} M. \033[0m'.format(param_num/1e6))

if torch.cuda.is_available():
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
    print('\033[1;31mUsing {} GPU(s) for training -----> \033[0m'.format(torch.cuda.device_count()))
    L2_pixelwise.cuda()
    L1_pixelwise.cuda()
########################################################################################################################
optimizer_G = torch.optim.Adam(denoise_generator.parameters(),
                                lr=opt.lr, betas=(opt.b1, opt.b2))

########################################################################################################################
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

prev_time = time.time()

########################################################################################################################
time_start=time.time()

# start training
def train_epoch():
    global prev_time
    denoise_generator.train()
    train_data = trainset(train_name_list, train_coordinate_list, train_noise_img, stack_index)
    trainloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    total_loss_list = []

    for iteration, noisy in enumerate(trainloader):

        noisy = noisy.cuda()
        mask1, mask2, mask3 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        noisy_sub3 = generate_subimages(noisy, mask3)
        
        
        noisy_output = denoise_generator(noisy_sub1)

        loss2neighbor_1 = 0.5*L1_pixelwise(noisy_output, noisy_sub2) + 0.5*L2_pixelwise(noisy_output, noisy_sub2)
        loss2neighbor_2 = 0.5*L1_pixelwise(noisy_output, noisy_sub3) + 0.5*L2_pixelwise(noisy_output, noisy_sub3)

        ################################################################################################################
        optimizer_G.zero_grad()
        # Total loss
        Total_loss = 0.5 * loss2neighbor_1 + 0.5 * loss2neighbor_2
        Total_loss.backward()
        optimizer_G.step()
        ################################################################################################################
        batches_done = epoch * len(trainloader) + iteration
        batches_left = opt.n_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
        prev_time = time.time()

        total_loss_list.append(Total_loss.item())

        if iteration % 1 == 0:
            time_end = time.time()
            print(
                '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f] [ETA: %s] [Time cost: %.2d s] '
                % (
                    epoch + 1,
                    opt.n_epochs,
                    iteration + 1,
                    len(trainloader),
                    np.mean(total_loss_list),
                    time_left,
                    time_end - time_start
                ), end=' ')

        if (iteration + 1) % len(trainloader) == 0:
            print('\n', end=' ')

        ################################################################################################################
        # save model
        if (iteration + 1) % (len(trainloader)) == 0:
            model_save_name = pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(
                4) + '.pth'
            if isinstance(denoise_generator, nn.DataParallel):
                torch.save(denoise_generator.module.state_dict(), model_save_name)  # parallel
            else:
                torch.save(denoise_generator.state_dict(), model_save_name)  # not parallel


for epoch in range(0, opt.n_epochs):
    train_epoch()
