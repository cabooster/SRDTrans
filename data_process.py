import numpy as np
import os
import tifffile as tiff
from skimage import io
import random
import math
import torch
from torch.utils.data import Dataset
from skimage import io


def random_transform(input):
    p_trans = random.randrange(8)  # (64, 128, 128)
    if p_trans == 0:  # no transformation
        input = input
    elif p_trans == 1:  # left rotate 90
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_trans == 2:  # left rotate 180
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_trans == 3:  # left rotate 270
        input = np.rot90(input, k=3, axes=(1, 2))
    elif p_trans == 4:  # horizontal flip
        input = input[:, :, ::-1]
    elif p_trans == 5:  # horizontal flip & left rotate 90
        input = input[:, :, ::-1]
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_trans == 6:  # horizontal flip & left rotate 180
        input = input[:, :, ::-1]
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_trans == 7:  # horizontal flip & left rotate 270
        input = input[:, :, ::-1]
        input = np.rot90(input, k=3, axes=(1, 2))
    return input

class Masker():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='zero'):
        self.window_size = width
        self.mode = mode

    def mask(self, X):

        mask = self.create_mask(X)
        mask = mask.to(X.device)

        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        if self.mode == 'interpolate':
            masked = self.interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        else:
            raise NotImplementedError


        return masked, mask

    def create_mask(self, input):
        mask = torch.zeros(input.shape)
        phase_x, phase_y = np.random.randint(0, 3, 2)
        for frame_id in range(input.shape[0]):
            mask[frame_id] = self.generate_single_mask(input[frame_id].shape, phase_x, phase_y, self.window_size)
        return torch.Tensor(mask)

    def generate_single_mask(self, shape, phase_x, phase_y, phase_size):
        cur_mask = torch.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (i % phase_size == phase_x) and (j % phase_size == phase_y):
                    cur_mask[i, j] = 1
        return cur_mask

    def interpolate_mask(self, tensor, mask, mask_inv):

        device = tensor.device

        mask = mask.to(device)

        kernel = np.array(
            [
                [[0.5, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 0.5]],
                [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                [[0.5, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 0.5]],
            ]
        )
        kernel = kernel[np.newaxis, np.newaxis, :, :, :]
        kernel = torch.Tensor(kernel).to(device)
        kernel = kernel / kernel.sum()

        filtered_tensor = torch.nn.functional.conv3d(tensor, kernel, stride=1, padding=1)

        return filtered_tensor * mask + tensor * mask_inv

class Denormalize(object):
    def __init__(self, min_pixel, max_pixel, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

    def __call__(self, data):
        data =  data.cpu().detach().numpy().astype(np.float32)
        data = self.std * data + self.mean
        data = np.clip(data, 0, 1)
        data = data * (self.max_pixel - self.min_pixel) + self.min_pixel
        return data


class trainset(Dataset):
    def __init__(
            self, name_list, coordinate_list,
            noise_img_all, stack_index
    ):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index

    def __getitem__(self, index):
        # fn = self.images[index]
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        input = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]


        input = random_transform(input)
        
        input = torch.from_numpy(np.expand_dims(input, 0).copy())

        return input

    def __len__(self):
        return len(self.name_list)


class testset(Dataset):
    def __init__(self,name_list,coordinate_list,noise_img):
        self.name_list = name_list
        self.coordinate_list=coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        #fn = self.images[index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch=torch.from_numpy(np.expand_dims(noise_patch, 0))
        #target = self.target[index]
        return noise_patch, single_coordinate

    def __len__(self):
        return len(self.name_list)


def get_gap_t(args, img, stack_num):
    whole_x = img.shape[2]
    whole_y = img.shape[1]
    whole_t = img.shape[0]
    #print('whole_x -----> ',whole_x)
    #print('whole_y -----> ',whole_y)
    #print('whole_t -----> ',whole_t)
    w_num = math.floor((whole_x-args.patch_x)/args.gap_x)+1
    h_num = math.floor((whole_y-args.patch_y)/args.gap_y)+1
    s_num = math.ceil(args.train_datasets_size/w_num/h_num/stack_num)
    # print('w_num -----> ',w_num)
    # print('h_num -----> ',h_num)
    # print('s_num -----> ',s_num)
    gap_t = math.floor((whole_t-args.patch_t)/(s_num-1))
    #gap_t = math.floor((whole_t)/(s_num-1))
    # print('gap_t -----> ',gap_t)
    return gap_t


def train_preprocess_lessMemoryMulStacks(args):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t = args.patch_t
    gap_y = args.gap_y
    gap_x = args.gap_x
    # gap_t2 = args.gap_t*2
    im_folder = os.path.join(args.datasets_path, args.datasets_folder)

    name_list = []
    coordinate_list={}
    stack_index = []
    noise_im_all = []
    ind = 0
    print('\033[1;31mImage list for training -----> \033[0m')
    print('All files are in -----> ', im_folder)
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    print('Total stack number -----> ', stack_num)

    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print(im_name)
        im_dir = os.path.join(im_folder, im_name)
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>args.select_img_num:
            noise_im = noise_im[0:args.select_img_num,:,:]
        gap_t = get_gap_t(args, noise_im, stack_num)


        assert gap_y >= 0 and gap_x >= 0 and gap_t >= 0, "train gat size is negative!"
        # args.gap_t = gap_t
        # print('gap_t -----> ', gap_t)
        # print('gap_x -----> ', gap_x)
        # print('gap_y -----> ', gap_y)

        noise_im = noise_im.astype(np.float32) / args.scale_factor  # no preprocessing
        noise_im = noise_im-noise_im.mean()
        
        noise_im_all.append(noise_im)
        
        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_t = noise_im.shape[0]
        for x in range(0,int((whole_y-patch_y+gap_y)/gap_y)):
            for y in range(0,int((whole_x-patch_x+gap_x)/gap_x)):
                for z in range(0,int((whole_t-patch_t+gap_t)/gap_t)):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    init_h = gap_y*x
                    end_h = gap_y*x + patch_y
                    init_w = gap_x*y
                    end_w = gap_x*y + patch_x
                    init_s = gap_t*z
                    end_s = gap_t*z + patch_t
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind = ind + 1
    return name_list, noise_im_all, coordinate_list, stack_index


def singlebatch_test_save(single_coordinate, output_image, raw_image):
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    aaaa = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    bbbb = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def multibatch_test_save(single_coordinate,id,output_image,raw_image):
    stack_start_w_id = single_coordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w'].numpy()
    stack_end_w=int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w'].numpy()
    patch_start_w=int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w'].numpy()
    patch_end_w=int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate['stack_start_h'].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate['stack_end_h'].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate['patch_end_h'].numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = single_coordinate['stack_start_s'].numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = single_coordinate['stack_end_s'].numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = single_coordinate['patch_start_s'].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = single_coordinate['patch_end_s'].numpy()
    patch_end_s = int(patch_end_s_id[id])

    output_image_id=output_image[id]
    raw_image_id=raw_image[id]
    aaaa = output_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    bbbb = raw_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return aaaa,bbbb,stack_start_w,stack_end_w,stack_start_h,stack_end_h,stack_start_s,stack_end_s


def test_preprocess_lessMemoryNoTail_chooseOne (args, N):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t2 = args.gap_t
    cut_w = (patch_x - gap_x)/2
    cut_h = (patch_y - gap_y)/2
    cut_s = (patch_t2 - gap_t2)/2

    assert cut_w >=0 and cut_h >= 0 and cut_s >= 0, "test cut size is negative!"
    im_folder = os.path.join(args.datasets_path, args.datasets_folder)

    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    # print(img_list)

    im_name = img_list[N]

    im_dir = os.path.join(im_folder, im_name)
    noise_im = tiff.imread(im_dir)
    
    input_data_type = noise_im.dtype
    img_mean = noise_im.mean()
    
    if noise_im.shape[0]>args.test_datasize:
        noise_im = noise_im[0:args.test_datasize,:,:]
    noise_im = noise_im.astype(np.float32)/args.scale_factor
    noise_im = noise_im-img_mean
    # noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.scale_factor

    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]

    num_w = math.ceil((whole_x-patch_x+gap_x)/gap_x)
    num_h = math.ceil((whole_y-patch_y+gap_y)/gap_y)
    num_s = math.ceil((whole_t-patch_t2+gap_t2)/gap_t2)

    for z in range(0, num_s):
        for x in range(0,num_h):
            for y in range(0,num_w):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_y*x
                    end_h = gap_y*x + patch_y
                elif x == (num_h-1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w-1):
                    init_w = gap_x*y
                    end_w = gap_x*y + patch_x
                elif y == (num_w-1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s-1):
                    init_s = gap_t2*z
                    end_s = gap_t2*z + patch_t2
                elif z == (num_s-1):
                    init_s = whole_t - patch_t2
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y*gap_x
                    single_coordinate['stack_end_w'] = y*gap_x+patch_x-cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x-cut_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_x-patch_x+cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y*gap_x+cut_w
                    single_coordinate['stack_end_w'] = y*gap_x+patch_x-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x-cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x*gap_y
                    single_coordinate['stack_end_h'] = x*gap_y+patch_y-cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y-cut_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_y-patch_y+cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x*gap_y+cut_h
                    single_coordinate['stack_end_h'] = x*gap_y+patch_y-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y-cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z*gap_t2
                    single_coordinate['stack_end_s'] = z*gap_t2+patch_t2-cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2-cut_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_t-patch_t2+cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2
                else:
                    single_coordinate['stack_start_s'] = z*gap_t2+cut_s
                    single_coordinate['stack_end_s'] = z*gap_t2+patch_t2-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2-cut_s

                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list, img_mean, input_data_type
