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
parser.add_argument("--n_epochs", type=int, default=40, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0,1', help="the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')")

parser.add_argument('--patch_x', type=int, default=150, help="the width of 3D patches (patch size in x)")
parser.add_argument('--patch_t', type=int, default=150, help="the width of 3D patches (patch size in t)")
parser.add_argument('--overlap_factor', type=float, default=0.5, help="the overlap factor between two adjacent patches")

parser.add_argument('--batch_size', type=int, default=1, help="the batch_size)")

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
parser.add_argument('--fmap', type=int, default=16, help='number of feature maps')

parser.add_argument('--scale_factor', type=int, default=1, help='the factor for image intensity scaling')
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--datasets_folder', type=str, default='train', help="A folder containing files for training")
parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
parser.add_argument('--key_word', type=str, default='', help="pth file root path")
parser.add_argument('--clean_img_path', type=str, required=True, help="path of clean img")

parser.add_argument('--select_img_num', type=int, default=1000000, help='select the number of images used for training (how many slices)')
parser.add_argument('--train_datasets_size', type=int, default=4000, help='datasets size for training (how many patches)')
parser.add_argument('--test_datasize', type=int, default=1000, help='datasets size for test (how many frames)')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU

#############################################################################################################################################
from SRDTrans import SRDTrans
from data_process import train_preprocess_lessMemoryMulStacks, trainset, test_preprocess_lessMemoryNoTail_chooseOne, testset, singlebatch_test_save, multibatch_test_save
from utils import save_yaml_train
from sampling import *


def cal_snr(noise_img, clean_img):
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal ** 2
    clean_signal_2 = clean_signal ** 2
    sum1 = np.sum(clean_signal_2)
    sum1 = 1e-10 if sum1 == 0 else sum1
    sum2 = np.sum(noise_signal_2)
    sum2 = 1e-10 if sum2 == 0 else sum2
    snrr = 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))
    return snrr


def cal_tif_snr(noisy_tif, clean_tif):
    total_num = clean_tif.shape[0]
    assert clean_tif.shape[0] == noisy_tif.shape[0], "length of clean and noisy don't match"
    snr_list = []
    for idx in range(total_num):
        snr = cal_snr(noisy_tif[idx], clean_tif[idx])
        snr_list.append(snr)
    print("final snr is:", np.mean(snr_list))


def cal_tif_snr_list(noisy_tif, clean_tif):
    noisy_tif = np.squeeze(noisy_tif)
    clean_tif = np.squeeze(clean_tif)
    total_num = clean_tif.shape[0]
    assert clean_tif.shape[0] == noisy_tif.shape[0], "length of clean and noisy don't match"
    snr_list = []
    for idx in range(total_num):
        snr = cal_snr(noisy_tif[idx], clean_tif[idx])
        snr_list.append(snr)
    return snr_list

########################################################################################################################
# use isotropic patch size by default
opt.patch_y = opt.patch_x   # the height of 3D patches (patch size in y)
opt.patch_t = opt.patch_t   # the length of 3D patches (patch size in t)
opt.gap_x = int(opt.patch_x*(1-opt.overlap_factor))   # patch gap in x
opt.gap_y = int(opt.patch_y*(1-opt.overlap_factor))   # patch gap in y
opt.gap_t = int(opt.patch_t*(1-opt.overlap_factor))  # patch gap in t
# opt.ngpu = [int(ts) for ts in opt.GPU.split(',')]                  # check the number of GPU used for training
opt.ngpu = opt.GPU.count(',')+1
opt.batch_size = opt.batch_size                          # By default, the batch size is equal to the number of GPU for minimal memory consumption
print('\033[1;31mTraining parameters -----> \033[0m')
print(opt)

########################################################################################################################
if not os.path.exists(opt.output_dir): 
    os.mkdir(opt.output_dir)
current_time = opt.datasets_folder+'_'+opt.key_word+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
output_path = opt.output_dir + '/' + current_time
pth_path = 'pth//'+ current_time
print("ckp is saved in {}".format(pth_path))
if not os.path.exists(pth_path): 
    os.mkdir(pth_path)

train_name_list, train_noise_img, train_coordinate_list, stack_index = train_preprocess_lessMemoryMulStacks(opt)
test_name_list, test_noise_img, test_coordinate_list = test_preprocess_lessMemoryNoTail_chooseOne(opt, N=0)
clean_img = tiff.imread(opt.clean_img_path).astype(np.float32)

yaml_name = pth_path+'//para.yaml'
save_yaml_train(opt, yaml_name)
########################################################################################################################

L1_pixelwise = torch.nn.L1Loss()
L2_pixelwise = torch.nn.MSELoss()

denoise_generator = SRDTrans(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=256,
    num_heads=8,
    hidden_dim=128*4,
    window_size=7,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

param_num = sum([param.nelement() for param in denoise_generator.parameters()])
print('\033[1;31mParameters of model is {:.2f}M. \033[0m'.format(param_num/1e6))

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
    train_data = trainset(train_name_list, train_coordinate_list, train_noise_img, stack_index, clean_img=clean_img)
    trainloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    snrr2clean_list = []
    snrr2target_list = []
    total_loss_list = []

    for iteration, (noisy, clean) in enumerate(trainloader):

        noisy = noisy.cuda()
        mask1, mask2, mask3 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        noisy_sub3 = generate_subimages(noisy, mask3)
        
        clean_sub1 = generate_subimages(clean, mask1)
        
        noisy_output = denoise_generator(noisy_sub1)
        noisy_target = noisy_sub2

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

        target_numpy = noisy_target.cpu().detach().numpy().astype(np.float32)
        output_numpy = noisy_output.cpu().detach().numpy().astype(np.float32)
        clean_numpy = clean_sub1.cpu().detach().numpy().astype(np.float32)
        snrr2target = cal_snr(output_numpy[0:opt.patch_t, :, :], target_numpy[0:opt.patch_t, :, :])
        snrr2clean = cal_snr(output_numpy[0:opt.patch_t, :, :], clean_numpy[0:opt.patch_t, :, :])

        snrr2clean_list.append(snrr2clean)
        snrr2target_list.append(snrr2target)

        total_loss_list.append(Total_loss.item())

        if iteration % 1 == 0:
            time_end = time.time()
            print(
                '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f] [ETA: %s] [Time cost: %.2d s] [SNR2target: %.2f dB] [SNR2clean: %.2f dB] '
                % (
                    epoch + 1,
                    opt.n_epochs,
                    iteration + 1,
                    len(trainloader),
                    np.mean(total_loss_list),
                    time_left,
                    time_end - time_start,
                    np.mean(snrr2target_list),
                    np.mean(snrr2clean_list)
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


def valid(pth_index):
    denoise_generator.eval()
    denoise_img = np.zeros(test_noise_img.shape)

    global_snr_list = []
    test_data = testset(test_name_list, test_coordinate_list, test_noise_img, clean_img=clean_img[:opt.test_datasize, :, :])
    testloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    with torch.no_grad():
        for iteration, (noise_patch, clean_patch, single_coordinate) in enumerate(testloader):
            noise_patch = noise_patch.cuda()
            real_A = noise_patch
            real_A = Variable(real_A)
            fake_B = denoise_generator(real_A)

            preditc_numpy = fake_B.cpu().detach().numpy().astype(np.float32)
            clean_numpy = clean_patch.detach().numpy().astype(np.float32)
            global_snr_list += cal_tif_snr_list(preditc_numpy, clean_numpy)

            ################################################################################################################
            # Determine approximate time left
            batches_done = iteration
            batches_left = 1 * len(testloader) - batches_done
            ################################################################################################################
            if iteration % 1 == 0:
                print(
                    '\r[Model %d] [Patch %d/%d] [SNR: %.6f]     '
                    % (
                        pth_index,
                        iteration + 1,
                        len(testloader),
                        np.mean(global_snr_list)
                    ), end=' ')

            if (iteration + 1) % len(testloader) == 0:
                print('\n', end=' ')
            ################################################################################################################
            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())
            if (output_image.ndim == 3):
                turn = 1
            else:
                turn = output_image.shape[0]
            # print(turn)
            if (turn > 1):
                for id in range(turn):
                    # print('shape of output_image -----> ',output_image.shape)
                    aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                        single_coordinate, id, output_image, raw_image)
                    denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5
            else:
                aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                    single_coordinate, output_image, raw_image)
                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5

        # del noise_img
        output_img = denoise_img.squeeze().astype(np.float32) * opt.scale_factor
        del denoise_img
        output_img = output_img.astype('int16')
        cal_tif_snr(output_img, clean_img[:opt.test_datasize, :, :])


for epoch in range(0, opt.n_epochs):
    train_epoch()
    valid(epoch)